## Runs the model for the team of Tropf, Newman, Geurgas, Thompson, Sheppard, Wright

# run.R can be used to test your submission.

# List your packages here. Don't forget to update packages.R!

require(psych)
require(rpart)
require(plyr)
require(stringi)
require(caret)


clean_df <- function(df, background_df = NULL){
  # Preprocess the input dataframe to feed the model.
  ### If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command
  
  # Parameters:
  # df (dataframe): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
  # background (dataframe): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).
  
  # Returns:
  # data frame: The cleaned dataframe with only the necessary columns and processed variables.
  
  Train_full<-df
  
  obvious_fr<-data.frame(nomem_encr= Train_full$nomem_encr, 
                         # Age = Train_full$cf20m004,
                         Age = Train_full$age_bg,
                         #   Gender = Train_full$cf20m003, 
                         Gender = Train_full$gender_bg,
                         N_new_kids = Train_full$cf20m129, 
                         t_new_kids = Train_full$cf20m130, 
                         have_partner = Train_full$cf20m024, 
                         cohabit_partner = Train_full$cf20m025,
                         gender_partner = Train_full$cf20m032,
                         samesex_code = c(Train_full$cf20m003+Train_full$cf20m032), ## zero is homosexual MM, 1 is hetero, 2 is homosexual FF
                         partner_age = Train_full$cf20m026, 
                         year_partnered = Train_full$cf20m028,
                         Smokes_past = Train_full$ch20m125,
                         Smokes_now = Train_full$ch20m126, 
                         Drinks_now = Train_full$ch20m133, 
                         Drinks_week = Train_full$ch20m134,
                         ## did they give up drinking this year?
                         Quit_drink = c(Train_full$ch20m133+c(!(Train_full$ch20m134==FALSE))), 
                         brutoink_f_2020 = Train_full$brutoink_f_2020,
                         woonvorm_2020 = Train_full$woonvorm_2020,
                         burgstat_2020 = Train_full$burgstat_2020,
                         SRH = Train_full$ch20m004)
  
  ## in-fills missing ages
  obvious_fr$Age[is.na(obvious_fr$Age)]<-(c(2020-Train_full$birthyear_bg[is.na(Train_full$age_bg)])[is.na(obvious_fr$Age)])
  
  ## infills the rest with LOCF
  obvious_fr$brutoink_f_2020[which(is.na(obvious_fr$brutoink_f_2020))]<-Train_full$brutoink_f_2019[is.na(obvious_fr$brutoink_f_2020)]
  obvious_fr$burgstat_2020[which(is.na(obvious_fr$burgstat_2020))]<-Train_full$burgstat_2019[is.na(obvious_fr$burgstat_2020)]
  obvious_fr$woonvorm_2020[which(is.na(obvious_fr$woonvorm_2020))]<-Train_full$woonvorm_2019[is.na(obvious_fr$woonvorm_2020)]
  
  obvious_fr$t_new_kids_2019<-Train_full$cf19l130
  
  obvious_fr$t_new_kids_2<-obvious_fr$t_new_kids
  obvious_fr$t_new_kids_2[is.na(obvious_fr$t_new_kids)]<-c(Train_full$cf19l130[is.na(obvious_fr$t_new_kids)]-1)
  
  ## infills non-answers as -9
  obvious_fr$t_new_kids_3<-obvious_fr$t_new_kids_2
  obvious_fr$t_new_kids_3[is.na(obvious_fr$t_new_kids_3)]<-c(-9)
  
  
  ## adds a variable for non-response rates
  #obvious_fr$n_missing<-rowSums(is.na(obvious_fr))
  obvious_fr$n_missing<-rowMeans(is.na(obvious_fr))
  
  obvious_fr$n_missing2020<-rowMeans(is.na(Train_full[,grep("^[a-z][a-z]20",colnames(Train_full))]))
  
  ## and for "want new kids" response rate
  obvious_fr$answer_kids<-is.na(obvious_fr$t_new_kids)
  
  ## builds a compound variable - 'kid affirmative'
  ## variable is -1 if they answered and don't plan on kids within a year
  ## variable is 0 if they did not answer
  ## variable is 1 if they answered and plan on kids within a year
  obvious_fr$comp_kid<-c(as.numeric(obvious_fr$answer_kids)-1)
  obvious_fr$comp_kid[which(obvious_fr$t_new_kids<=1)]<-1
  
  ## and for "drinks now" response rate
  obvious_fr$answer_drink<-is.na(obvious_fr$Drinks_now)
  
  obvious_fr$comp_drink<-c(as.numeric(obvious_fr$answer_drink)-1)
  obvious_fr$comp_drink[which(obvious_fr$answer_drink==1)]<-1
  
  ## Hotsauce! -pew pew pew-
  df <-obvious_fr
  
  return(df)
}

predict_outcomes <- function(df, background_df = NULL, model_path = "./model.rds"){
  # Generate predictions using the saved model and the input dataframe.
  
  # The predict_outcomes function accepts a dataframe as an argument
  # and returns a new dataframe with two columns: nomem_encr and
  # prediction. The nomem_encr column in the new dataframe replicates the
  # corresponding column from the input dataframe The prediction
  # column contains predictions for each corresponding nomem_encr. Each
  # prediction is represented as a binary value: '0' indicates that the
  # individual did not have a child during 2021-2023, while '1' implies that
  # they did.
  
  # Parameters:
  # df (dataframe): The data dataframe for which predictions are to be made.
  # background_df (dataframe): The background data dataframe for which predictions are to be made.
  # model_path (str): The path to the saved model file (which is the output of training.R).
  
  # Returns:
  # dataframe: A dataframe containing the identifiers and their corresponding predictions.
  
  ## This script contains a bare minimum working example
  if( !("nomem_encr" %in% colnames(df)) ) {
    warning("The identifier variable 'nomem_encr' should be in the dataset")
  }
  
  # Load the model
  model <- readRDS(model_path)
  
  # Preprocess the fake / holdout data
  df <- clean_df(df, background_df)
  
  # Exclude the variable nomem_encr if this variable is NOT in your model
#  vars_without_id <- colnames(df)[colnames(df) != "nomem_encr"]
  
  # Generate predictions from model
  predictions <- predict(model, df, type = "class") 
  
  predictions<-as.numeric(as.character(predictions))
  
  # Create predictions that should be 0s and 1s rather than, e.g., probabilities
#  predictions <- ifelse(predictions > 0.5, 1, 0)  
  
  # Output file should be data.frame with two columns, nomem_encr and predictions
  df_predict <- data.frame("nomem_encr" = df$nomem_encr, "prediction" = predictions)
  
  # Force columnnames (overrides names that may be given by `predict`)
  names(df_predict) <- c("nomem_encr", "prediction") 
  
  # Return only dataset with predictions and identifier
  return( df_predict )
}
