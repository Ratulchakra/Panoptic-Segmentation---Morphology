# Panoptic-Segmentation---Morphology

Here we give project update


```
h, w = 2D shape of Skeleton Map
Set, k = 0
Get, New = Empty Image with shape (h, w)
Get, Semantic Mask = Skeleton Map > ε (~ 0)
Get, object_boundary = Dilation of Semantic Mask - Semantic Mask
Get, Object_centroid = Skeleton Map > δ (~ 1)

function flood_fill (x, y, k) {
    If x > h or y > w or x < 0 or y < 0:
        return
    Else:
        If object_boundary(x, y) == 0 and New(x, y) == 0:
            New (x, y) = k
            flood_fill(x+1, y, k)
            flood_fill(x-1, y, k)
            flood_fill(x, y+1, k)
            flood_fill(x, y-1, k)
}

for (centroid in Object_centroid){
    x, y = centroid
    k ++
    flood_fill(x, y, k)
}
```
