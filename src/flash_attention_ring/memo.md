1: init an float array to -inf, memset will not work. memset is good at setting all the bytes in the memory to be exactly the same. using a simple kernel to work on it.

2: warp op will fall into infinite loop if the warp are defined in a wrong way.

3: the online softmax will accumulate some loss. because when we perform float computing: (x/y) \* (y/z) !== x/z
