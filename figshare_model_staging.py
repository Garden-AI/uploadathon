import modal

iris_image = (
    modal.Image.from_registry("python:3.11-bullseye")
    .apt_install("wget")
    .pip_install("scikit-learn==1.2.2", "pandas==2.1.2")
    .run_commands("wget -O model.joblib https://figshare.com/ndownloader/files/51847034")
)

app = modal.App("iris-example-figshare")

@app.function(image=iris_image)
def predict_iris_type(input_array):
    import pandas as pd
    import joblib
    
    # Load the model into memory.
    model = joblib.load(f"/model.joblib")

    # Run the model on the user-provided input.
    input_as_df = pd.DataFrame(input_array, columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']) 
    predictions = model.predict(input_as_df)
    as_strings = [['setosa', 'versicolor', 'virginica'][prediction] for prediction in predictions]
    print(as_strings)
    return as_strings

@app.local_entrypoint()
def test_iris_model():
    example_input = [
        [5.5, 2.4, 3.7, 1. ],
        [6.3, 2.8, 5.1, 1.5],
        [6.4, 3.1, 5.5, 1.8],
        [6.6, 3. , 4.4, 1.4],
        [5.1, 3.5, 1.4, 0.2],
    ]
    result = predict_iris_type.remote(example_input)
    print(result)
    return result