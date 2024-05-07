import React from 'react';

const Base64Image = ({ base64String, alt, format }) => {
  // Determine the image format based on the 'format' prop
  const imageFormat = format || 'png';
  const decodedImage = `data:image/${imageFormat};base64,${base64String}`;

  return (
    <div>
      {/* <h1>Base64 Image Example</h1> */}
      <img src={decodedImage} alt={alt} />
    </div>
  );
};

export default Base64Image;
