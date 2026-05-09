# Failure Case Analysis

Strategy: Hybrid (BM25 + Semantic) | Threshold: 0.3 | Queries: 20

## Correct Refusals
_Guardrail correctly refused - low confidence AND answer would be poor_
Count: 0/20

## Incorrect Refusals (False Positives)
_Guardrail refused but the system could have given a reasonable answer_
Count: 0/20

## Faithful Answers
_Guardrail passed and the answer is well-grounded in context_
Count: 4/20

### Example 1
**Query**: why do police need to understand what the forensic scientists do
**Reference**: No Answer Present.
**Generated**: The context does not provide information to answer why police need to understand what forensic scientists do.
**Confidence**: 0.5000 | **ROUGE-L**: 0.100 | **Faithfulness**: 0.500
**Top Retrieved Chunk** (score=0.5000):
> Doctors do not understand why most cases of early onset Alzheimer's appear at such a young age. But in a few hundred families worldwide, scientists have pinpointed several rare genes that directly cau

### Example 2
**Query**: average number of lightning strikes per day
**Reference**: Globally 8,640,000 lightning strikes per day.
**Generated**: There is no information provided in the context about the average number of lightning strikes per day.
**Confidence**: 0.5000 | **ROUGE-L**: 0.320 | **Faithfulness**: 0.417
**Top Retrieved Chunk** (score=0.5000):
> According to our estimations, daily change rates of Rwanda population in 2015 will be the following: 1  1 253 live births average per day (52.19 in a hour). 2  337 deaths average per day (14.04 in a h

### Example 3
**Query**: are volvox autotrophic or heterotrophic
**Reference**: No Answer Present.
**Generated**: According to [2], Volvox are protists, and many protists, such as algae, are photosynthetic. Therefore, Volvox are autotrophic.
**Confidence**: 0.6725 | **ROUGE-L**: 0.000 | **Faithfulness**: 0.455
**Top Retrieved Chunk** (score=0.6725):
> Some organisms rely on organic compounds as a source of carbon, but are able to use light or inorganic compounds as a source of energy. Such organisms are not defined as autotrophic, but rather as het

## Unfaithful Answers (False Negatives)
_Guardrail passed but the answer is poorly grounded_
Count: 16/20

### Example 1
**Query**: . what is a corporation?
**Reference**: A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.
**Generated**: The context does not provide information to answer the question about what a corporation is.
**Confidence**: 0.5000 | **ROUGE-L**: 0.162 | **Faithfulness**: 0.091
**Top Retrieved Chunk** (score=0.5000):
> What are the Essential Elements of Job Descriptions? . . . . . . . . . . . . . . . . . . . . . . 2. What are Task Statements? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

### Example 2
**Query**: why did rachel carson write an obligation to endure
**Reference**: Rachel Carson writes The Obligation to Endure because believes that as man tries to eliminate unwanted insects and weeds, however he is actually causing more problems by polluting the environment.
**Generated**: There is no information provided in the context about why Rachel Carson wrote "An Obligation to Endure."
**Confidence**: 0.5682 | **ROUGE-L**: 0.213 | **Faithfulness**: 0.308
**Top Retrieved Chunk** (score=0.5682):
> In saying this, I'm not like those others you complain about: you know damn well I understand what you did and why you had to do it. But just the same, there were some things that you did not have to 

### Example 3
**Query**: why did the progressive movement fail to advance racial equality quizlet
**Reference**: No Answer Present.
**Generated**: The context provided does not contain information to answer the question about why the progressive movement failed to advance racial equality on Quizlet.
**Confidence**: 0.5985 | **ROUGE-L**: 0.077 | **Faithfulness**: 0.235
**Top Retrieved Chunk** (score=0.5985):
> But because of the court’s earlier decisions, everything has to be stuffed into the “diversity” box. So universities say they want to advance diversity but never explain what they mean (nor did the co

## Summary

| Category | Count | Rate |
|----------|-------|------|
| Correct Refusals | 0 | 0% |
| Incorrect Refusals | 0 | 0% |
| Faithful Answers | 4 | 20% |
| Unfaithful Answers | 16 | 80% |