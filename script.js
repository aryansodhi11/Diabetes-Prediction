// script.js

document
  .getElementById("diabetes-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();

    // Collect data from form fields
    const data = {
      Pregnancies: document.getElementById("Pregnancies").value,
      Glucose: document.getElementById("Glucose").value,
      BloodPressure: document.getElementById("BloodPressure").value,
      SkinThickness: document.getElementById("SkinThickness").value,
      Insulin: document.getElementById("Insulin").value,
      BMI: document.getElementById("BMI").value,
      DiabetesPedigreeFunction: document.getElementById(
        "DiabetesPedigreeFunction"
      ).value,
      Age: document.getElementById("Age").value,
    };

    // Send a POST request to the /predict endpoint
    const response = await fetch("https://8388-144-48-178-201.ngrok-free.app/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    // Display the result
    const resultElement = document.getElementById("result");
    const result = await response.json();

    // Check the prediction and display the corresponding message
    let predictionMessage;
    if (result.prediction === 1) {
      predictionMessage = "Diabetic";
    } else if (result.prediction === 0) {
      predictionMessage = "Non-Diabetic";
    } else {
      predictionMessage = "Unknown"; // Handle other cases if needed
    }

    resultElement.innerText = `Prediction: ${predictionMessage}`;
  });
