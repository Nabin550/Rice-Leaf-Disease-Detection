// import { useState } from "react";
// import axios from "axios";

// // const class_labels_resnet = ['Bacterial leaf blight', 'brown spot', 'leaf smut'];
// const class_labels_cnn = ['Bacterial leaf blight', 'brown spot', 'leaf smut'];
// const class_labels_svm = ['Bacterial leaf blight', 'brown spot', 'leaf smut'];

// function Upload() {
//   const [image, setImage] = useState('');
//   const [predictedClass, setPredictedClass] = useState(null);
//   const [predictedClass1, setPredictedClass1] = useState(null);
//   const [predictedClass2, setPredictedClass2] = useState(null);
//   const [c1, setc1] = useState(null);
//   const [c2, setc2] = useState(null);
//   const [c3, setc3] = useState(null);
 

//   const handleChange = (e) => {
//     console.log(e.target.files)
//     setImage(e.target.files[0])
//   };

//   const handleApi = () => {
//     //call the api
//     const url = 'http://127.0.0.1:5000/';

//     const formData = new FormData();
//     formData.append('image', image);

//     axios.post(url, formData)
//     .then(result => {
//       console.log(result.data);
//       const { predicted_class_rf, predicted_class_cnn, predicted_class_svm } = result.data;

//         // Handle predicted_class_rf
//         const predictedClassArray = Array.isArray(predicted_class_rf) ? predicted_class_rf : [];
//         const [className, probability] = predictedClassArray.length === 2 ? predictedClassArray : ['Unable to predict', null];

//         setPredictedClass({ className, probability });

//       setPredictedClass1(result.data.predicted_class_cnn);
//       setPredictedClass2(result.data.predicted_class_svm);
//       setc1(result.data.c1);
//       setc2(result.data.c2);
//       setc3(result.data.c3);
      
//       console.log(result.data.predicted_class_rf);
//       console.log(result.data.predicted_class_cnn);
//       console.log(result.data.predicted_class_svm);
//       // console.log(result.data)
//       alert('Image uploaded sucessfully');
//     })
//       .catch(error => {
//         alert('service error');
//         console.log(error);
//       });
//   }

//   return (
//     <div className="uploadcontainer">
//     <div className="uploadimage">
//     <span>Upload Image:</span>
//       <input  type="file" id="choosefile" onChange={handleChange}  />
//       <button className="button" onClick={handleApi}>SUBMIT</button>
//     </div>

//     {predictedClass && typeof predictedClass === 'object' && (
//         <div className="output-container">
//           <div className="output">
//             <span className="predict">Predicted Class(rf): </span>
//             {predictedClass.className} with probability {predictedClass.probability}
//           </div>
//         </div>
//       )}

//     {predictedClass1 !==null && (
//         <div className="output-containercnn">
//           {/* <div className="output"><span className="predict">Predicted Class(resnet): </span> {predictedClass}</div> */}
//           <div className="outputcnn"><span className="predict_cnn">Predicted Class(cnn): </span> {class_labels_cnn[predictedClass1]}</div>
//         </div>
//     )}
//     {predictedClass2 !==null && (
//         <div className="output-containercnn">
//           {/* <div className="output"><span className="predict">Predicted Class(resnet): </span> {predictedClass}</div> */}
//           <div className="outputsvm"><span className="predict_cnn">Predicted Class(svm): </span> {class_labels_svm[predictedClass2]}</div>
//         </div>
//     )}



//     </div>
//   );
// }

// export default Upload;