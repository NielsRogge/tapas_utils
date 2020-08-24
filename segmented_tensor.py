import torch
from torch_scatter import scatter

class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """Creates an index.
        Args:
            indices: torch.LongTensor of indices, same shape as `values`.
            num_segments: Scalar tensor, the number of segments. All elements
            in a batched segmented tensor must have the same number of segments
            (although many segments can be empty).
            batch_dims: Python integer, the number of batch dimensions. The first
            `batch_dims` dimensions of a SegmentedTensor are treated as batch
            dimensions. Segments in different batch elements are always distinct
            even if they have the same index.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return self.indices.size()[:self.batch_dims] # returns a torch.Size object


class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """Combines indices i and j into pairs (i, j).
        The result is an index where each segment (i, j) is the intersection of
        segments i and j. For example if the inputs represent table cells indexed by
        respectively rows and columns the output will be a table indexed by
        (row, column) pairs, i.e. by cell.
        The implementation combines indices {0, .., n - 1} and {0, .., m - 1} into
        {0, .., nm - 1}. The output has `num_segments` equal to
            `outer_index.num_segments` * `inner_index.num_segments`.
        Args:
        outer_index: IndexMap.
        inner_index: IndexMap, must have the same shape as `outer_index`.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
         raise ValueError('outer_index.batch_dims and inner_index.batch_dims must be the same.')

        super(ProductIndexMap, self).__init__(
            indices=(inner_index.indices +
                    outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims)
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=(index.indices // self.inner_index.num_segments).type(torch.float).floor().type(torch.long),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims
        )

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments).type(torch.float).floor().type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims
        )

def gather(values, index, name='segmented_gather'):
    """Gathers from `values` using the index map.
    For each element in the domain of the index map this operation looks up a
    value for that index in `values`. Two elements from the same segment always
    get assigned the same value.
    Args:
        values: [B1, ..., Bn, num_segments, V1, ...] Tensor with segment values.
        index: [B1, ..., Bn, I1, ..., Ik] IndexMap.
        name: Name for the TensorFlow operation.
    Returns:
        [B1, ..., Bn, I1, ..., Ik, V1, ...] Tensor with the gathered values.
    """
    print("Values:")
    print(values)
    print("Index:")
    print(index.indices)
    indices = index.indices
    # first, check whether the indices of the index represent scalar values (i.e. not vectorized)
    if len(values.shape[index.batch_dims:]) < 2:
        return torch.gather(values, 
                        index.batch_dims, 
                        indices.view(values.size()[0], -1) # torch.gather expects index to have the same number of dimensions as values
                       ).view(indices.size())
    else:
        # this means we have a vectorized version
        # we have to adjust the index
        indices = indices.unsqueeze(-1).expand(values.shape)
        return torch.gather(values, index.batch_dims, indices)
    #out = torch.zeros(index.indices.size())
    #return values.squeeze(index.batch_dims)[index.indices]

def flatten(index, name='segmented_flatten'):
    """Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map.
    This operation relabels the segments to keep batch elements distinct. The k-th
    batch element will have indices shifted by `num_segments` * (k - 1). The
    result is a tensor with `num_segments` multiplied by the number of elements
    in the batch.
    Args:
        index: IndexMap to flatten.
        name: Name for the TensorFlow operation.
    Returns:
        The flattened IndexMap.
    """
    # first, get batch_size as scalar tensor
    batch_size = torch.prod(torch.tensor(list(index.batch_shape()))) 
    # next, create offset as 1-D tensor of length batch_size,
    # and multiply element-wise by num segments (to offset different elements in the batch) e.g. if batch size is 2: [0, 64]
    offset = torch.arange(start=0, end=batch_size) * index.num_segments  
    offset = offset.view(index.batch_shape()) 
    for _ in range(index.batch_dims, len(index.indices.size())): # typically range(1,2)
        offset = offset.unsqueeze(-1)

    indices = offset + index.indices
    return IndexMap(
        indices=indices.view(-1),
        num_segments=index.num_segments * batch_size,
        batch_dims=0)

def range_index_map(batch_shape, num_segments, name='range_index_map'):
    """Constructs an index map equal to range(num_segments).
    Args:
        batch_shape: torch.Size object indicating the batch shape 
        num_segments: int, indicating number of segments
        name: Name for the TensorFlow operation.
    Returns:
        IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = torch.as_tensor(batch_shape, dtype=torch.int64) # create a rank 1 tensor vector containing batch_shape (e.g. [2]) 
    assert len(batch_shape.size()) == 1
    num_segments = torch.as_tensor(num_segments) # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    assert len(num_segments.size()) == 0

    indices = torch.arange(start=0, end=num_segments) # create a rank 1 vector with num_segments elements 
    new_tensor = torch.cat([
        torch.ones_like(batch_shape, dtype=torch.long),
        num_segments.unsqueeze(dim=0)
        ],
        dim=0)
    # new_tensor is just a vector of [1 64] for example (assuming only 1 batch dimension)
    new_shape = [int(x) for x in new_tensor.tolist()]
    indices = indices.view(new_shape) 
    
    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    indices = indices.repeat(multiples.tolist()) 
    # equivalent (in Numpy:)
    #indices = torch.as_tensor(np.tile(indices.numpy(), multiples.tolist()))
    
    return IndexMap(
        indices=indices,
        num_segments=num_segments,
        batch_dims=list(batch_shape.size())[0])

def _segment_reduce(values, index, segment_reduce_fn, name):
    """Applies a segment reduction segment-wise."""
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()):] # torch.Size object
    flattened_shape = torch.cat([torch.as_tensor([-1],dtype=torch.int64), torch.as_tensor(vector_shape, dtype=torch.int64)], dim=0)
    flat_values = values.view(flattened_shape.tolist())

    # currently:
    # flat_index.indices = torch.FloatTensor
    # flat_values = torch.LongTensor

    # we make src and out should be FloatTensors,
    # and index.indices should be LongTensor

    #print(flat_index.indices)
    #print(flat_values)

    segment_means = scatter(src=flat_values, index=flat_index.indices.type(torch.int64), dim=0, reduce=segment_reduce_fn)

    #print(segment_means)
    
    #torch.zeros(18, 1).scatter_add(1, flat_index.indices.unsqueeze(-1), flat_values.unsqueeze(-1))
    
    #print(torch.as_tensor(vector_shape, dtype=torch.int64))
    #print(torch.as_tensor(index.batch_shape()))
    #print(torch.as_tensor(index.num_segments, dtype=torch.int64))
    
    # Unflatten the values.
    new_shape = torch.cat(
        [torch.as_tensor(index.batch_shape(), dtype=torch.int64), 
        torch.as_tensor([index.num_segments], dtype=torch.int64),
        torch.as_tensor(vector_shape, dtype=torch.int64)],
        dim=0)
    #print("New shape:")
    #print(new_shape)
    
    output_values = segment_means.view(new_shape.tolist())
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    #return None
    return output_values, output_index

def reduce_sum(values, index, name='segmented_reduce_sum'):
    """Sums a tensor over its segments.
    Outputs 0 for empty segments.
    This operations computes the sum over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in
        a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present
        the output will be a sum of vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
        index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
        name: Name for the TensorFlow ops.
    Returns:
        A pair (output_values, output_index) where `output_values` is a tensor
        of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..] and `index` is an
        IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "sum", name)

def reduce_mean(values, index, name='segmented_reduce_mean'):
    """Averages a tensor over its segments.
    Outputs 0 for empty segments.
    This operations computes the mean over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in
        a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present
        the output will be a mean of vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
        index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
        name: Name for the TensorFlow ops.
    Returns:
        A pair (output_values, output_index) where `output_values` is a tensor
        of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..] and `index` is an
        IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "mean", name)

def reduce_max(values, index, name='segmented_reduce_max'):
    """Computes the maximum over segments.
    This operations computes the maximum over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in
        a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present
        the output will be an element-wise maximum of vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
        index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
        name: Name for the TensorFlow ops.
    Returns:
        A pair (output_values, output_index) where `output_values` is a tensor
        of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..] and `index` is an
        IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "max", name)






    
