#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getMatrixPlot = function(mat, style, prefix = NULL) {

  checkmate::assertChoice(x=style, choices=c("boxplot", "violin"), .var.name="style") 

  df = na.omit(melt(mat))
  colnames(df)[1] = "algo"
  
  expression = "\\.classif|.preproc|.tuned|.model_selection|._search"
  df$algo = gsub(x = df$algo, pattern =  expression, replacement = "")

  g = ggplot(data = df, mapping = aes(x = as.factor(algo), y = value))

  if(style == "boxplot") {
    g = g + stat_boxplot(geom ='errorbar')
    g = g + geom_boxplot(outlier.colour = "black", outlier.size = 0.5)
  } else if(style == "violin") {
    g = g + geom_violin(trim = TRUE, scale = "width", fill = "darkgrey")
    g = g + geom_boxplot(outlier.colour = "black", outlier.size = 0.5, width = 0.1, fill = "white")
  }
 
  g = g + theme(text = element_text(size = 10), 
    axis.text.x = element_text(angle = 0, vjust = .5, hjust = 1))
  g = g + ylab(paste("% of the maximum", prefix)) + xlab("Algorithms")
  g = g + scale_y_continuous(limits = c(0, 1))
  g = g + coord_flip()

  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getMatrixHeatMap = function(mat) {

  mat$task = paste0("OML_Task_",rownames(mat))
  df = melt(mat, id.vars = ncol(mat))
  colnames(df) = c("task", "algo", "percentage")

  expression = "\\.classif|.preproc|.tuned|.model_selection|._search"
  df$algo = gsub(x = df$algo, pattern =  expression, replacement = "")
  df$task = gsub(x = df$task, pattern = "OML_Task_", replacement = "")

  g = ggplot(df, aes(x = task, y = as.factor(algo), fill=percentage, colour=percentage))
  g = g + geom_tile() + theme_bw()
  g = g + scale_fill_gradient(low = "white", high = "red", na.value = "black", limits = c(0,1))
  g = g + scale_colour_gradient(low = "white", high = "red", na.value = "black", limits = c(0,1))
  g = g + theme(text = element_text(size = 10))
  g = g + xlab("OpenML task ids") + ylab("Algorithms")
  g = g + theme(axis.text.x = element_text(angle = 90, vjust = .5, hjust = 1, size = 5))
 
  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------