#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

getFailedBarPlot = function(mat) {

  algo = colnames(mat)
  N = sapply(mat, function(elem) length(which(is.na(elem))))
  df = data.frame(cbind(algo, N))
  rownames(df) = NULL

  expression = "\\.classif|.preproc|.tuned|.model_selection|._search"
  df$algo = gsub(x = df$algo, pattern = expression, replacement = "")
  df$algo = factor(df$algo, levels = df$algo[order(df$N, decreasing = TRUE)])
  df$N    = as.numeric(as.character(df$N))

  g = ggplot(df, mapping = aes(x = algo, y = N, fill = N))
  g = g + geom_bar(stat = "identity")
  g = g + scale_y_continuous(limits = c(0,100))
  g = g + scale_fill_continuous(high = "red", low = "black", guide = )
  g = g + xlab('Algorithm') + ylab('Number of jobs that failed')
  g = g + coord_flip() + guides(fill = FALSE)
  g = g + theme_classic()

  # TODO: add information about why they failed
  return(g)
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
