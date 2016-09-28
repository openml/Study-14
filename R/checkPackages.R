#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

checkPackages = function(pkgs) {

  obj = installed.packages()

  for(pk in pkgs) {
    
    if(pk %in% rownames(obj)) {
      cat(paste0(" - Package: ", pk, " \t... is already installed\n"))
    } else {
      cat(paste0(" - Installing: ", pk, "\n"))  
      if (pk == "farff") {
        devtools::install_github("mlr-org/farff")
      } else if(pk == "OpenML") {
        devtools::install_github("openml/r", ref = "05b8b97cc5ce6ea1b3f586818cfcf157b16a3cd4")
      } else {
        install.packages(pkgs = pk)   
      }
    }
  }
}

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------