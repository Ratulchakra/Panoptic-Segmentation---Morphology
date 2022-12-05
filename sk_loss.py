import torch 
from kornia.morphology import dilation,erosion
import torchvision.transforms as transforms
import torch.nn.functional as F


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
NUM_CLASSES = 21



#instance_gt shape=(num_inst,H,W)
#this function return skeleton map of each instance . output shape is same as input
def  find_sk_map(instance_gt,kernel_shape=(3,3)):
    N,H,W=instance_gt.shape
    instance_gt = instance_gt.unsqueeze(1)  #shape=(#num_inst,1,H,W)
    instance_gt= transforms.Pad(1)(instance_gt)
    kernel=torch.ones(kernel_shape).to(instance_gt.device)

    num_iteration=int(max(W,H))

    temp=instance_gt
    count=[]
    temp_list=[]
    temp_list.append(instance_gt)
    count.append(torch.sum(temp,axis=(1,2,3))>0)  # What's happening?
    for i in range(num_iteration):
        temp=erosion(temp,kernel)
        temp_count=torch.sum(temp,axis=(1,2,3))>0
        if(torch.sum(temp_count)<=0):
            break
        temp_list.append(temp)
        count.append(temp_count)

    count=torch.stack(count)
    count=count.type(torch.float)

    temp_list=torch.stack(temp_list)
    temp_list=temp_list[:,:,:,1:-1,1:-1]
   
    
    Sk_map=torch.zeros((N,H,W)).to(device) 
    for n in range(N):
        temp_sk=torch.tensordot(temp_list[:,n],count[:,n],dims=([0],[0]))/torch.sum(count[:,n])
        Sk_map[n]=temp_sk[0]      

    return(Sk_map)
         

#sk_map size=(N,H,W)
def find_sk_center(sk_map,kernel_shape=(3,3)):
    sk_map=sk_map.unsqueeze(1)
    kernel=torch.ones(kernel_shape).to(device)
    temp=sk_map
    temp=erosion(temp,kernel)
    #temp=erosion(temp,kernel)
    #temp=dilation(temp,kernel)
    temp=dilation(temp,kernel)

    out=sk_map-temp
    out=out.squeeze(1)
    return(out)




#seg mask gt shape =(512, 512)  same as inst_mask_gt with all gt labels in integers
def find_gt_skmaps(seg_mask_gt,inst_mask_gt):
      seg_unique=torch.unique(seg_mask_gt)
      inst_unique=torch.unique(inst_mask_gt)
      net_gt=torch.zeros((NUM_CLASSES,)+seg_mask_gt.shape).to(device)
      net_gt = net_gt.permute((1, 0, 2, 3))


      Inst_list=[]
      loc_list=[]
      for uniq in inst_unique:
          inst_single=(inst_mask_gt==uniq)
          inst_loc=torch.where(inst_single>0)
          idx,idy=torch.median(inst_loc[0]),torch.median(inst_loc[1]) #have to check idx ,idy
          seg_loc=seg_mask_gt[idx,idy]   # Why do this? It will only give the value at that location not the actual location
          Inst_list.append(inst_single)
        #   print(seg_loc)
          loc_list.append(seg_loc.long())  # This is not a list of locations, but of unique values
    
      Inst_tensor=torch.stack(Inst_list).type(torch.float)
      print(Inst_tensor.shape)
      print(loc_list[0].shape)
      print(len(loc_list))
      print(net_gt.shape)
    #   loc_list = loc_list[1:]

      #####################################
    #   Inst_tensor = Inst_tensor[1:]
      Inst_tensor = Inst_tensor.permute((1, 0, 2, 3))
      Inst_list = []
      for i in range(Inst_tensor.shape[0]):
        Inst_list.append(find_sk_map(Inst_tensor[i]))
      Inst_sk = torch.stack(Inst_list).type(torch.float)
    #   Inst_sk = Inst_sk.permute((1, 0, 2, 3))
      print(Inst_sk.shape)
      ###################################


    #   Inst_sk=find_sk_map(Inst_tensor)
      #print("***********",seg_mask_gt.device,inst_mask_gt.device,net_gt.device,Inst_tensor.device,Inst_sk.device)
      net_gt[:, loc_list]=Inst_sk
      return(net_gt)

##########################################################################

def gt_sk_map(seg_mask_gt,inst_mask_gt):
    # seg_unique=torch.unique(seg_mask_gt)
    # inst_unique=torch.unique(inst_mask_gt)
    net_gt=torch.zeros((NUM_CLASSES,)+seg_mask_gt.shape).to(device)
    net_gt = net_gt.permute((1,0,2,3))   # B, N, H, W
    b, n, _, _ = net_gt.shape

    # seg_mask_gt = torch.nn.functional.one_hot(seg_mask_gt.to(torch.int64), 21).permute((0, 3, 1, 2))

    for i in range(b):
        inst = inst_mask_gt[i]
        net = net_gt[i]
        print(net.shape)
        inst_unique=torch.unique(inst)
        print(inst_unique)
        inst_gt = torch.nn.functional.one_hot(inst.to(torch.int64), len(inst_unique)).permute((2, 0, 1))
        print(inst_gt.shape)
        seg_loc = []
        for j in range(inst_gt.shape[0]):
            inst_loc=torch.where(inst_gt[j]>0)
            idx,idy=torch.median(inst_loc[0]),torch.median(inst_loc[1])
            seg_loc.append(seg_mask_gt[i][idx,idy].long())
        print(seg_loc)
        inst_sk = find_sk_map(inst_gt)
        print(inst_sk.shape)
        for j in range(inst_sk.shape[0]):
            net[seg_loc[j]] = inst_sk[j]
    print(net_gt.shape)
    return net_gt

#########################################################################







def  sk_loss(outs, target_seg,target_inst):
    batch_size,H,W=target_seg.shape
    loss=0
    for b_idx in range(batch_size):
        seg_mask_gt=target_seg[b_idx]
        inst_mask_gt=target_inst[b_idx]
        net_gt_mask=find_gt_skmaps(seg_mask_gt,inst_mask_gt)
        for out in outs:
            temp=net_gt_mask.unsqueeze(0)
            #temp=net_gt_mask
            #print(temp.shape,out.shape[1:])
            new_gt_out = F.interpolate(temp,(out.shape[1],out.shape[2]))#,mode='linear')
            new_gt_out=new_gt_out.squeeze(0)
            I=(new_gt_out>0).type(torch.float)
        
            out1= out*I
            out0=(1-out)*(1-I)

            gt1=new_gt_out
            gt0=(1-new_gt_out)*(1-I)

      
            loss=loss+torch.sum(torch.square(out1-gt1))+torch.sum(torch.square(out0-gt0))
            #loss=loss+torch.mean(torch.square(new_gt_out-out))
            #loss=loss+torch.mean(torch.abs(new_gt_out-out))

    return loss,net_gt_mask

def  sk_loss1(outs, target_seg,target_inst):
    #print(outs[1].shape)
    loss=torch.mean(outs)
    return(loss)
