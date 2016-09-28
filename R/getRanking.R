#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# Obs: Not handling NAs values (removed from the ranking)

getRanking = function(mat, descending = FALSE) {
  
  temp = mat
  for(i in 1:nrow(mat)) {
    ids =  which(!is.na(mat[i,]))
    if(descending){
      temp[i, ids] = rank(-mat[i,ids])
    } else {
      temp[i, ids] = rank( mat[i,ids])  
    }
  }

  # average ranking
  aux = lapply(1:ncol(temp), function(i) {
    ids = which(!is.na(temp[,i]))
    return( mean(temp[ids,i]) )
  })

  rk.mean = data.frame(do.call("rbind", aux))
  rk.mean$alg = colnames(temp)
  colnames(rk.mean) = c("rk.avg", "alg")
  rk.mean = rk.mean[, c(2,1)]
  ret = list(rk = temp, rk.avg = rk.mean)

  return(ret)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
