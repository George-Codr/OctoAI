# server.py
import uvicorn, asyncio, logging
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from brain import SuperBrain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("superbrain.server")

app = FastAPI(title="SuperBrain v2")
brain = SuperBrain()

class QueryIn(BaseModel):
    q: str
    stream: bool = False

@app.on_event("shutdown")
async def shutdown_event():
    await brain.client.close()

async def event_stream(query: str):
    # minimal streaming: yield partial messages (here: run full pipeline then chunk)
    res = await brain.ask(query)
    # first chunk: activated regions
    yield f"data: REGIONS {res['regions']}\n\n"
    # then each expert snippet
    for k,v in res["experts"].items():
        txt = v['text'][:1000].replace("\n", " ")
        yield f"data: EXPERT {k} {txt}\n\n"
    # finally final answer
    yield f"data: FINAL {res['final']}\n\n"
    yield "data: DONE\n\n"

@app.post("/ask")
async def ask(qin: QueryIn, background_tasks: BackgroundTasks):
    if qin.stream:
        return StreamingResponse(event_stream(qin.q), media_type="text/event-stream")
    else:
        res = await brain.ask(qin.q)
        return JSONResponse(res)

class FeedbackIn(BaseModel):
    query: str
    rating: float  # 0..1
    notes: str = ""

@app.post("/feedback")
async def feedback(fb: FeedbackIn):
    # TODO: wire feedback to Bandit/Exp3 selector (left as simple append)
    logger.info("Feedback received for %s: rating=%s", fb.query, fb.rating)
    # Here we could map the feedback to update selectors (future work)
    return {"status":"ok"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
