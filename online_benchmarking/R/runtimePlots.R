#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getRuntimePlot = function(data, style) { 

  if(!(style %in% c("point", "boxplot", "violin"))) {
    stop("Please, provide a valid style: point, boxplot or violin ")
  }

  temp = dplyr::select(.data = data, flow.name, usercpu.time.millis.training, 
    usercpu.time.millis.testing, usercpu.time.millis)

  df = melt(temp, id.vars = 1)
  colnames(df) = c("algo", "measure", "value")

  g = ggplot(data = df, mapping = aes(x = as.factor(algo), y = log(value), fill = measure))
  if(style == "point") {
    g = g + geom_point(aes(colour = measure)) 
  } else if(style == "boxplot") {
    g = g + stat_boxplot(geom ='errorbar')  
    g = g + geom_boxplot(outlier.colour = "black", outlier.size = 0.5) 
  } else if(style == "violin") {
    g = g + geom_violin(trim = TRUE, scale = "width")
    g = g + geom_boxplot(outlier.colour = "black", outlier.size = 0.5, width = 0.2, fill = "white")
  }
  g = g + facet_grid(. ~ measure)
  g = g + ylab("log(time)") + xlab("Algorithms")
  g = g + theme(legend.position="none")
  g = g + scale_colour_brewer(palette="Dark2")
  g = g + coord_flip()

  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
