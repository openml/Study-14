#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

.auxYLabel = function(measure) {

  if(measure == "usercpu.time.millis") {
    y.label = "log(Runtime)"
  } else if(measure == "usercpu.time.millis.training") {
    y.label = "log(Training time)"
  } else if(measure == "usercpu.time.millis") {
    y.label = "log(Testing time)"
  } else {
    y.label = gsub(measure, pattern="\\.", replacement=" ")
  }
  return(y.label)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


getSimplePlot = function(data, measure = "predictive.accuracy", style, landscape = TRUE, 
  prefix = NULL) {

  checkmate::assertChoice(x=measure, choices=AVAILABLE.MEASURES, .var.name="measure") 
  checkmate::assertChoice(x=style, choices=c("boxplot", "violin"), .var.name="style") 
  
  temp = data[, c("flow.name", measure)]
  colnames(temp) = c("algo", "meas")

  expression = "\\.classif|.preproc|.tuned|.model_selection|._search"
  temp$algo  = gsub(x = temp$algo, pattern =  expression, replacement = "")

  y.label = .auxYLabel(measure = measure)
  if(!is.null(prefix)) {
    y.label = paste(prefix, y.label)
  }

  if(measure %in% c("usercpu.time.millis",  "usercpu.time.millis.training", 
    "usercpu.time.millis.testing")) { 
    g = ggplot(data = temp, mapping = aes(x = as.factor(algo), y = log(meas)))
  } else {
    g = ggplot(data = temp, mapping = aes(x = as.factor(algo), y = meas))
    g = g + scale_y_continuous(limits = c(0, 1))
  }

  g = g + theme_bw()

  if(style == "violin") {
    g = g + geom_violin(trim = TRUE, scale = "width", fill = "darkgray")
    g = g + geom_boxplot(outlier.colour = "black", outlier.size = 0.5, width = 0.2, 
      fill = "white")
  } else {
    g = g + stat_boxplot(geom ='errorbar')
    g = g + geom_boxplot(outlier.colour = "black", outlier.size = 0.5, fill = "darkgray")
  }

  g = g + theme(legend.position="none")
  g = g + xlab("Algorithms") + ylab(y.label)
  
  if(landscape) {
    g = g + coord_flip()
  } else {
    g = g + theme(text = element_text(size = 10), axis.text.x = element_text(angle = 90, 
      vjust = .5, hjust = 1))
  }
  
  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------