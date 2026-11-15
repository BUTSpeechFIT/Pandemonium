"""Muse DIARIZATION event stream -> segments. A turn spans speechStart..speechEnd (audioProcessedMs) with the
speechComplete transcript; a `speaker` label covers audio since the previous speechStart/speaker event, so a turn
takes the label(s) emitted inside it -- several labels inside one turn split its words evenly in time."""
def segments(events):
    segs, cur, labels = [], None, []
    for e in events:
        t = e.get("type")
        if t == "speechStart":
            cur, labels = {"start": e["audioProcessedMs"] / 1000}, []
        elif t == "speaker" and cur is not None:
            labels.append(e["label"])
        elif t == "speechEnd" and cur is not None:
            cur["end"] = e["audioProcessedMs"] / 1000
        elif t == "speechComplete" and cur is not None:
            words = (e.get("transcript") or "").split()
            s, en = cur["start"], cur.get("end", e["audioProcessedMs"] / 1000)
            labs = labels or ["?"]
            n = len(labs)
            for k, lab in enumerate(labs):
                part = words[k * len(words) // n:(k + 1) * len(words) // n]
                if part:
                    segs.append({"start": s + (en - s) * k / n, "end": s + (en - s) * (k + 1) / n,
                                 "speaker": lab, "text": " ".join(part)})
            cur = None
    return segs
