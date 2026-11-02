# ARC: Autonomous Reasoning Companion

The robot's mind: an offline language model on the Jetson that tells heritage stories, answers
visitors in Bahasa Melayu, English or German, and decides when the arm should perform or look.

```
arc/
  persona.md          who ARC is and how it speaks (always in the prompt)
  topics/NAME.yaml    one per sub-topic: role, audience, which knowledge cards, opening line
  knowledge/*.md      fact cards, ONE topic each, checked by the team (verified: true + source)
  eval.yaml           test questions: languages, facts, unknowns, off-topic, actions
```

## How it answers (cache-augmented generation)

The whole topic's knowledge pack (a few thousand tokens) sits in the system prompt and stays in
the model's KV cache (llama-server keeps the shared prefix). So every question sees every fact,
nothing has to be searched, and only the new question costs time. Facts come ONLY from the cards:
if the answer is not in them, ARC says it is not sure (a wrong date in front of judges costs
more than "I don't know").

## Rules for knowledge cards

- One heritage item per card, short sentences, plain facts (the model retells them in any language).
- `verified: false` until a team member has checked every sentence against a source; put the
  source in `source:`. `iro arc check` lists unverified cards.
- Never copy a fact from memory or an AI answer without a source: even published papers disagree
  (e.g. whether Wayang Kulit Kelantan is on a UNESCO list).

## Day 1 sub-topic switch (about 5 minutes)

1. `cp arc/topics/heritage.yaml arc/topics/NEWTOPIC.yaml`, edit role, focus and the card list
2. `iro arc topic NEWTOPIC`
3. `iro arc chat` and ask it three questions about the sub-topic
