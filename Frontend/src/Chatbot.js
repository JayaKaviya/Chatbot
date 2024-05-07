import React, { useState } from 'react';
import Base64Image from './Base64Image'; // Adjust the path based on your project structure
import './Chatbot.css'; // Import a CSS file for styling

const Chatbot = () => {
  const [userInput, setUserInput] = useState('');
  const [conversation, setConversation] = useState([]); 

  const handleUserInput = (event) => {
    setUserInput(event.target.value);
  };

  const handleSendMessage = async () => {
    try {
      // ... (rest of the code remains the same)

                const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_input: userInput }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`); 
      }

      const responseData = await response.json();

      // Update the conversation with the user's input and the response
      setConversation([
        ...conversation,
        { type: 'user', text: userInput },
        { type: 'bot', text: responseData.response, image: responseData.predicted_image_base64 },
      ]);

      // Clear the user input
      setUserInput('');
    } catch (error) {
      console.error('Error sending message:', error.message);
    }
  };

  return (
    <div className="chatbot-container">
      <div className="chat-history">
        {/* Display conversation history */}
        {conversation.map((message, index) => (
          <div key={index} className={message.type}>
            {message.type === 'user' ? (
              <p>USER :  {message.text}</p>
            ) : (
              <div>
                <p>BOT :  {message.text}</p>
                {message.image && (
                   <div className="image-container">
                   {/* Use the Base64Image component and pass the image format */}
                   <Base64Image
                     base64String={message.image}
                     alt="No Image!"
                     format={message.image_format}
                     className="image"
                   />
                 </div>
           
                )}
              </div>
            )}
          </div>
        ))}
      </div>
      {/* Input for user */}
      <div className="user-input">
        <input type="text" value={userInput} onChange={handleUserInput} />
        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
};

export default Chatbot;






























