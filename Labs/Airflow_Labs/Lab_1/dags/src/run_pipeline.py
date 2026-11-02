from lab import load_data, data_preprocessing, build_save_model, load_model_evaluate

# run the four pipeline steps in order without the scheduler, for quick local testing
if __name__ == "__main__":
    data = load_data()
    preprocessed_data = data_preprocessing(data)
    results = build_save_model(preprocessed_data, 'wine_gmm_model.sav')
    result = load_model_evaluate('wine_gmm_model.sav', results)
    print(result)
