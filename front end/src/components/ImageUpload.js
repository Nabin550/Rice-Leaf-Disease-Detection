import { useState } from "react";
import axios from "axios";

function ImageUpload() {
  const [image, setImage] = useState('');
  const [predictedClass, setPredictedClass] = useState(null);
  const [score, setscore] = useState(null);
  const [predictedClass1, setPredictedClass1] = useState(null);
  const [score1, setscore1] = useState(null);
  const handleChange = (e) => {
    console.log(e.target.files)
    setImage(e.target.files[0])
  };

  const handleApi = () => {
    //call the api
    const url = 'http://127.0.0.1:5000/';

    const formData = new FormData();
    formData.append('image', image);

    axios.post(url, formData)
    .then(result => {
      setPredictedClass(result.data.predicted_class);
      setscore(result.data.c);
      setPredictedClass1(result.data.predicted_class1);
      setscore1(result.data.c1);
      // console.log(result.data)
      alert('Image uploaded sucessfully');
    })
      .catch(error => {
        alert('service error');
        console.log(error);
      });
  }

  return (
    <div className="uploadcontainer">
    <div className="uploadimage">
    <span>Upload Image:</span>
      <input  type="file" id="choosefile" onChange={handleChange}  />
      <button className="button" onClick={handleApi}>SUBMIT</button>
    </div>

    {predictedClass && (
        <div className="output-container">
          <div className="output"><span className="predict">Predicted Class(SVM): </span>{predictedClass}  prob : {score.toFixed(2)}</div>
            <div className="output1"><span className="predict">Predicted Class(RF): </span>{predictedClass1}  prob : {score1.toFixed(2)}</div>
          {/* <p className="predicted-output">{predictedClass}</p> */}

        </div>
    )}
    </div>
  );
}

export default ImageUpload;