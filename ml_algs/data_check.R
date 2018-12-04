loadhdf5data <- function(h5File, dataset) {
    require(h5) # available on CRAN
    
    f <- h5file(h5File)
    nblocks <- h5attr(f[dataset], "nblocks")
    
    data <- do.call(cbind, lapply(seq_len(nblocks)-1, function(i) {
        data <- as.data.frame(f[paste0(dataset, "/block", i, "_values")][])
        colnames(data) <- f[paste0(dataset, "/block", i, "_items")][]
        data
    }))
    
    h5close(f)
    data
}