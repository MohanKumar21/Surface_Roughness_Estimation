from fastapi import FastAPI 
import uvicorn
app=FastAPI()

@app.route('/')
def fun(x):
    return "Hello World"
