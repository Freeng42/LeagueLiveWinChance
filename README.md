# LeagueLiveWinChance
Inspired by the "AWS-Win probabilty" shown on LOL-Esports broadcasts

## To just run the win probabilty tool:
Put both LiveWinChance.py and lol_model.pt in the same folder and run "LiveWinChance.py" while you are in a game.

Only works on games run on the same device as the programm

Since it uses the localAPI no RiotGames API-key is needed.


## if you want to make your own model:

requirements: You need your own API key, to make your own dataset from your games or any other person.
The code currently expects a .env with your API key in the same folder as "GetDataSet.py"

Run GetDataSet.py type in the Player you want to get data from and wait for it to finnish.
You should now have a file called "lol_dataset.csv" in your folder.
Now run lol_model.py to create your own lol_model.pt model file to continue with the steps above.


For any questions or comments feel free to reach out
