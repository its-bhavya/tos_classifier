import asyncio, aiohttp, pandas as pd, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT = RAW_DIR / "clauses_raw.csv"

LIST_URL   = "https://api.tosdr.org/service/v3/?page={page}"
DETAIL_URL = "https://api.tosdr.org/service/v2/?id={sid}"

CONCURRENCY = 8         # gentler on rate limits
MAX_RETRIES = 6


async def get_json(session, url, sem):
    delay = 2.0
    async with sem:
        for _ in range(MAX_RETRIES):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status == 429 or r.status >= 500:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 30)
                        continue
                    r.raise_for_status()
                    return await r.json()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
    return None


async def list_all_services(session, sem):
    services, page = [], 1
    while True:
        data = await get_json(session, LIST_URL.format(page=page), sem)
        batch = (data or {}).get("services") or []
        if not batch:
            break
        services.extend(batch)
        print(f"  listed page {page}: +{len(batch)} (total {len(services)})")
        page += 1
    return services


async def fetch_service(session, sem, s, counter, total):
    sid, sname = s["id"], s["name"]
    data = await get_json(session, DETAIL_URL.format(sid=sid), sem)
    counter[0] += 1
    i = counter[0]
    if data is None:
        print(f"  [{i}/{total}] FAIL {sname} ({sid})")
        return []
    points = ((data.get("parameters") or {}).get("points")) or []
    rows = []
    for p in points:
        case = p.get("case") or {}
        rows.append({
            "title":        p.get("title", ""),
            "label":        case.get("classification", ""),
            "service_id":   sid,
            "service_name": sname,
            "category":     case.get("title", ""),
        })
    if points or i % 500 == 0:
        print(f"  [{i}/{total}] {sname} (id={sid}): +{len(points)}")
    return rows


async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    conn = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=conn) as session:
        services = await list_all_services(session, sem)
        all_count = len(services)
        services = [s for s in services if s.get("is_comprehensively_reviewed")]
        total = len(services)
        print(f"Filtered to comprehensively reviewed: {total}/{all_count}")
        print(f"Fetching details for {total} services (concurrency={CONCURRENCY}) -> {OUT}")

        counter = [0]
        tasks = [fetch_service(session, sem, s, counter, total) for s in services]
        results = await asyncio.gather(*tasks)

    rows = [r for batch in results for r in batch]
    df = pd.DataFrame(rows)
    df["label"] = df["label"].replace("blocker", "bad")
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df)} rows to {OUT}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    asyncio.run(main())
