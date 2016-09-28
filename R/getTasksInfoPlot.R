#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# Color blind palette (Dark2)
# http://colorbrewer2.org/
# ['#1b9e77','#d95f02','#7570b3']

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getTasksInfoPlot = function(data) {

  tasks = unique(data$task.id)
 
  aux = lapply(tasks, function(task){
    sub = data[which(data$task.id == task), ]
    max.acc = max(sub$predictive.accuracy)
    max.auc = max(sub$area.under.roc.curve)
    maj.prop = max(sub$perMajClass)
    ret = c(max.acc, max.auc, maj.prop)
    return(ret)
  })

  # df = [task | max.acc | max.auc | % maj class ]
  df = data.frame(do.call("rbind", aux))
  colnames(df) = c("max_acc", "max_auc", "perc_maj")  
  df$tasks = tasks
  
  # sort increasing the % majclass (tasks)
  df = df[order(df$perc_maj, decreasing = FALSE),]
  df$tasks = factor(df$tasks, levels = df$tasks)

  df.final = melt(df, id.vars = 4)
  colnames(df.final) = c("task", "Measure", "value")
  df.final$task = as.numeric(df.final$task)
 
  g = ggplot(data=df.final, aes(x=task, y=value, group=Measure, colour=Measure, shape=Measure)) 
  g = g + geom_point() + scale_colour_brewer(palette = "Dark2")
  g = g + ylab(" Maximum value / Majority Class") + xlab("Tasks")
  g = g + scale_x_continuous(limits = c(1, nrow(df)))
 
  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
