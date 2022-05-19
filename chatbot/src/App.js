// import logo from './logo.svg';
import './App.css';
import React, { useState, useEffect, useRef } from 'react';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';


const Messages = ({ messages }) => {

  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages]);

  return (
    <div>
      {messages ? messages.map((m) => <p key={m[1]}>{m[0]}: {m[1]}</p>) : null}
      <div ref={messagesEndRef} />
    </div>
  )
}

function App() {

  
  const [message, setMessage] = useState([])
  const [recieved, setRecieved] = useState(true)
  async function postData(url = '', data = {}) {
    // Default options are marked with *
    const response = await fetch(url, {
      method: 'POST', // *GET, POST, PUT, DELETE, etc.
      mode: 'cors', // no-cors, *cors, same-origin
      cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
      headers: {
        'Content-Type': 'application/json'
        // 'Content-Type': 'application/x-www-form-urlencoded',
      },
      referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
      body: JSON.stringify(data) // body data type must match "Content-Type" header
    });
    return response.json(); // parses JSON response into native JavaScript objects
  }

  function sendRequest(text) {
    setRecieved(false)
    postData('http://127.0.0.1:5000/ask', {
      text: text
    }).then(data => {
      console.log(data);
      setMessage(oldArr => [...oldArr, ['Chatbot', data]])
      setRecieved(true)
    });
  }

  // make form to send to server
  function handleSubmit(event) {
    event.preventDefault();
    let messages = event.target.elements[0].value
    setMessage(oldArr => [...oldArr, ['User', messages]])
    console.log(message)
    sendRequest(event.target.elements[0].value);
    event.target.elements[0].value = '';
  }

  return (
    <div className="App">
      <header className="App-header">
        <h2>Chatbot</h2>

        <React.Fragment>
          <CssBaseline />
          <Container maxWidth="sm" fixed>
            <Box sx={{ bgcolor: '#cfe8fc', height: '60vh', overflow: 'auto' }}>
              <div className="pad">
                <Messages messages={message} />
                {!recieved ? <p>Loading...</p> : null}
              </div>
            </Box>
          </Container>
        </React.Fragment>
        <hr/>
        <div>
        <form onSubmit={e => handleSubmit(e)} style={{display: 'flex'}}>
          <div>
            <TextField id="outlined-basic" label="Message" color="primary" variant="outlined" placeholder='Enter Message Here'/>
          </div>
          <Button variant="contained" type="Submit">Send Message</Button>
        </form>
        </div>
        
      </header>
    </div>
  );
}

export default App;
