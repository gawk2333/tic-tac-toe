  import './App.css';
  import { useEffect,useState, useCallback } from 'react'
  import request from 'superagent'

  let initialState = [0,0,0,0,0,0,0,0,0]
  function App() {
    const [playerX, setPlayerX] = useState({
      label:'X',
      sign:1,
      type: 'human',
    })
    const [playerO, setPlayerO] = useState({
      label:'O',
      sign:-1,
      type: 'bot',
    })

    const [board, setBoard] = useState(initialState);
    const [currentPlayer, setCurrentPlayer] = useState(playerX)
    const [start,setStart] = useState(true)
    const [result,setResult] = useState("")

    const nextPlayer = useCallback(() => {
      if(start){
        if(currentPlayer===playerX){
          return playerO
        } else {
          return playerX
        }
      }
    },[currentPlayer, playerO, playerX, start])

    const checkWinner = useCallback((board) => {
      // Check rows
      if(!board.some(c => c === 0 )){
        setStart(false)
        setResult("Draw")
        return false
      }
      const row0 = board[0] + board[1] + board[2]
      const row1 = board[3] + board[4] + board[5]
      const row2 = board[6] + board[7] + board[8]
      const col0 = board[0] + board[3] + board[6]
      const col1 = board[1] + board[4] + board[7]
      const col2 = board[2] + board[5] + board[8]
      const diag1 = board[0] + board[4] + board[8]
      const diag2 = board[2] + board[4] + board[6]
      const list = [row0, row1,row2, col0, col1, col2, diag1, diag2]
      if(list.some(ele => ele === 3)){
        setStart(false)
        setResult("X wins!")
        return false
      }
      if(list.some(ele => ele === -3)){
        setStart(false)
        setResult("O wins!")
        return false
      }
      return true;
    },[])
    
    const handleSelected = useCallback((e,index) => {
      if(!start){
        return
      }
      if (currentPlayer.sign) {
        let newBoard = [...board]
        newBoard[index]=currentPlayer.sign
        setBoard(newBoard)
        let result = checkWinner(newBoard)
        if(result){
          setCurrentPlayer(nextPlayer())
        }
    }
    },[board, checkWinner, currentPlayer, nextPlayer, start])

    const makeDecision = useCallback(async () => {
      if(start){
      const result = await request.post('http://127.0.0.1:5000/predict')
                                .set("Access-Control-Allow-Origin", "*")
                                .send({
                                  player: currentPlayer,
                                  board_status:board
                                })
                                .then(res=>res.body)
      const action = result.action
      handleSelected("",action)
    }},[board, currentPlayer, handleSelected, start])

    const gameStart = () => {
      setBoard(initialState)
      setCurrentPlayer(playerX)
      setStart(true)
      setResult("")
    }

    useEffect(() => {
      if (playerX.type === 'human' || playerO.type === 'human') {
        if (currentPlayer.type === 'bot' && start) {
          // Make the bot's decision
          makeDecision(currentPlayer.sign);
    
          // After a 1-second delay, switch to the next player
          setTimeout(() => {
            setCurrentPlayer(nextPlayer(nextPlayer()));
          }, 1000);
        }
      } else if (playerX.type === 'bot' && playerO.type === 'bot' && start) {
        // For the case where both players are bots
        console.log("current", currentPlayer);
    
        // Make the bot's decision
        makeDecision(currentPlayer.sign);
    
        // After a 1-second delay, switch to the next player
        setTimeout(() => {
          setCurrentPlayer(nextPlayer(nextPlayer()));
        }, 1000);
      }
    }, [currentPlayer, makeDecision, nextPlayer, playerO.type, playerX.type, start]);
    



    const getBorderStatus = (index) => {
      let row = Math.floor(index / 3);
      let col = index % 3;
      let borderClass = ""
      if(row === 0 || row === 1){
          borderClass += " bbot"
      }
      if(row === 1 || row === 2){
          borderClass += " btop"
      }
      if(col === 0 || col === 1){
          borderClass += " brig"
      }
      if(col === 1 || col === 2){
          borderClass += " brlft"
      }
      return borderClass
    }

    const getCellType = (value,index) => {
      if (board[index]===1){
        return (<div className={`shape-wrapper ${getBorderStatus(index)}`}>
                  <div class="x-shape">
                  <div class="line"/>
                  <div class="line"/>
                </div></div>)
      } else if (board[index] === -1) {
        return <div className={`shape-wrapper ${getBorderStatus(index)}`}><div class="o-shape"/></div>
      } else {
        let row = Math.floor(index / 3);
        let col = index % 3;
        return  <div className={`Cell ${getBorderStatus(index)}`} key={`row:${row},column:${col}`} onClick={(e)=> handleSelected(e,index)}/>
      }
    }

    return (
      <div className="App">
        <header className="App-header">
          <h1>Tic Tac Toe</h1>
        </header>
        <div className='pannel'>
          <button onClick={gameStart}>Restart</button>
          <span>  X:</span>
          <select value={playerX.type} onChange={(e)=> setPlayerX({...playerX,type:e.target.value})}>
            <option value="human">human</option>
            <option value="bot">bot</option>
        </select>
        <span>  O:</span>
          <select value={playerO.type} onChange={(e)=> setPlayerO({...playerO,type:e.target.value})}>
            <option value="bot">bot</option>
            <option value="human">human</option>
        </select>
        <span style={{ color: 'red' }}>{result}</span>
        </div>
        <div className="App-content">
        <div className='Board'>
          {board.map((value,index) => getCellType(value,index))}
          </div>
        </div>
      </div>
    );
  }

  export default App;
