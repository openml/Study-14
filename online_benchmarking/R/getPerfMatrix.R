#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getPerfMatrix = function(data, measure = "predictive.accuracy", weighted = FALSE, w = 0.1) {

  checkMeasure(measure = measure)
  cat(paste0(" - Getting performance matrix for: ", measure, "\n"))
  
  all.learners = unique(data$flow.name)
  all.tasks = unique(data$task.id)

  temp = data
  mat = matrix(data = NA, nrow = length(all.tasks), ncol = length(all.learners), 
    dimnames = list(all.tasks, all.learners))
  
  if(weighted) {
    cat(paste0(" - Generating values weighted by the runtime: w = ", w, " \n"))
  }
 
  for(i in 1:nrow(temp)) { 
    row.id = which(all.tasks == temp[i,]$task.id)
    col.id = which(all.learners == temp[i,]$flow.name)

    if(weighted & measure != "usercpu.time.millis") { 
      mat[row.id, col.id] = temp[i, measure] - (log(1 + temp$usercpu.time.millis[i]) * w)
    } else { 
      mat[row.id, col.id] = temp[i, measure]
    }
  }
  
  # Removing algs with no execution (not being applied on all tasks)
  uniquelength = sapply(data.frame(mat), function(x) length(unique(x)))
  mat = subset(data.frame(mat), select = uniquelength > 1)
  mat[mat < 0] = 0

  return(mat)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------