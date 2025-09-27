import asyncio, json, logging, math, random, time
from client import UnifiedClient
from memory import PersistentMemory
from config import MODELS, PROVIDER_PRIORS, DATA_DIR
from typing import Dict, Any

logger = logging.getLogger("superbrain.brain")

class Exp3Selector:
    def __init__(self, choices, gamma=0.2):
        self.choices = choices
        self.gamma = gamma
        self.weights = {c: 1.0 for c in choices}

    def sample(self, k=1):
        total_w = sum(self.weights.values())
        probs = {c: (1 - self.gamma) * (self.weights[c] / total_w) + (self.gamma / len(self.choices)) for c in self.choices}
        items = list(self.choices)
        chosen = random.choices(items, weights=[probs[i] for i in items], k=k)
        return chosen, probs

    def update(self, choice, reward, prob):
        x = reward / prob
        self.weights[choice] *= math.exp((self.gamma * x) / len(self.choices))

class SuperBrain:
    def __init__(self):
        self.client = UnifiedClient()
        self.mem = PersistentMemory()
        region_providers = {}
        for region, models in MODELS.items():
            provs = list({p for (p,m) in models})
            if provs:
                region_providers[region] = Exp3Selector(provs)
        self.selectors = region_providers

    async def classify_regions(self, query: str):
        q = query.lower()
        regions = set()
        if any(w in q for w in ["code","python","implement","debug"]): regions.add("coding")
        if any(w in q for w in ["latest","today","news","recent"]): regions.add("search")
        if any(w in q for w in ["prove","why","logic","solve","theorem"]): regions.add("reasoning")
        if any(w in q for w in ["story","poem","write","creative","imagine"]): regions.add("creativity")
        if regions:
            return list(regions)
        messages = [{"role":"system","content":"Return one or more of: reasoning, search, coding, creativity (comma-separated)."},
                    {"role":"user","content":query}]
        try:
            prov, model = MODELS["sensory"]
            resp = await self.client.chat(prov, model, messages, max_tokens=32)
            text = self._extract(resp).lower()
            chosen = [r.strip() for r in text.replace("and",",").split(",") if r.strip() in MODELS.keys()]
            return chosen or ["reasoning","search"]
        except Exception as e:
            logger.warning("Sensory failure: %s", e)
            return ["reasoning","search"]

    def _extract(self, response: Dict[str,Any]) -> str:
        try:
            if "choices" in response:
                return response["choices"][0]["message"]["content"]
            if "candidates" in response:
                return response["candidates"][0]["content"]["parts"][0]["text"]
            return json.dumps(response)
        except Exception:
            return str(response)

    async def run_region_experts(self, regions, query):
        tasks = []
        for region in regions:
            messages = [{"role":"system","content":"You are an expert for this region."},{"role":"user","content":query}]

            if region == "search":
                async def web_search():
                    prov, model = MODELS["search"][0]
                    prompt = [{"role":"system","content":"Return top 15 results with title, snippet, and URL in JSON format."},
                              {"role":"user","content":query}]
                    resp = await self.client.chat(prov, model, prompt, max_tokens=1500)
                    text = self._extract(resp)
                    return ("search_ai", model, "search", text)
                tasks.append(asyncio.create_task(web_search()))
                continue

            if region == "reasoning":
                async def reasoning_call():
                    ds_model = "DeepSeek-V3"
                    resp = await self.client.chat("deepseek", ds_model, messages, max_tokens=600)
                    text = self._extract(resp)
                    return ("deepseek", ds_model, "reasoning", text)
                tasks.append(asyncio.create_task(reasoning_call()))
                continue

            selector = self.selectors.get(region)
            if selector:
                chosen, probs = selector.sample(k=1)
                provider = chosen[0]
            else:
                provider = MODELS.get(region, [("deepseek","DeepSeek-V3")])[0][0]

            model = None
            for p,m in MODELS.get(region, []):
                if p==provider:
                    model = m
                    break
            if model is None:
                p,m = MODELS.get(region,[("deepseek","DeepSeek-V3")])[0]
                provider, model = p,m

            async def call(p=provider, m=model, reg=region):
                res = await self.client.chat(p, m, messages, max_tokens=600)
                return (p,m,reg,res)

            tasks.append(asyncio.create_task(call()))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        responses = {}
        for r in results:
            if isinstance(r, Exception):
                continue
            p,m,reg,res = r
            text = self._extract(res)
            responses[f"{reg}:{p}/{m}"] = {"text": text, "provider": p, "model": m}
        return responses

    async def verify_claims(self, query, final_text, expert_responses):
        def overlap(a,b):
            sa = set(a.lower().split())
            sb = set(b.lower().split())
            if not sa or not sb: return 0.0
            return len(sa & sb)/min(len(sa), len(sb))
        texts = [v["text"] for v in expert_responses.values()]
        n = len(texts)
        if n <=1: return {"agreement":1.0}
        total = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i+1,n):
                total += overlap(texts[i], texts[j])
                pairs += 1
        return {"agreement": total/pairs if pairs>0 else 0.0}

    async def aggregate(self, query, expert_responses):
        parts = []
        for k,v in expert_responses.items():
            prov = v["provider"]
            weight = PROVIDER_PRIORS.get(prov,0.8)
            parts.append(f"----\nSource: {k}\nWeight: {weight}\nText:\n{v['text']}\n")
        prompt = f"User question: {query}\nExpert outputs:\n{''.join(parts)}\nTask: produce ONE concise answer with TL;DR."
        messages = [{"role":"system","content":"You are an expert aggregator."},{"role":"user","content":prompt}]
        for prov, model in MODELS["summarizer"]:
            try:
                resp = await self.client.chat(prov, model, messages, max_tokens=900, temperature=0.2)
                return self._extract(resp)
            except Exception:
                continue
        return "\n\n".join([f"{k}\n{v['text']}" for k,v in expert_responses.items()])

    async def ask(self, query):
        recalls = self.mem.search(query, k=3)
        context = "\n".join([f"Recall: {t}" for t,meta,d in recalls]) if recalls else ""
        prompt_query = (context + "\n\n" + query).strip()
        regions = await self.classify_regions(query)
        expert_responses = await self.run_region_experts(regions, prompt_query)
        final = await self.aggregate(query, expert_responses)
        verify = await self.verify_claims(query, final, expert_responses)
        try:
            self.mem.add([f"Q: {query}\nA: {final}"], [{"time": time.time()}])
        except Exception:
            pass
        return {"regions": regions, "experts": expert_responses, "final": final, "verify": verify}
