# WIL5
Repo for WIL5 
# Data Preparation
The following steps are taken to generate synthetic data that mirrors the structure and characteristics of the real data you would expect to encounter.
## Step 1: Define the Data Schema

| Entity ID | Annual Clients | Infraction Type | Infraction Timeline | Public Complaints | Sentiment Analysis | Inspection Results | Total Risk Score |
|-----------|----------------|-----------------|---------------------|-------------------|---------------------|--------------------|------------------|
| E0001     | 150            | Minor           | Within past year    | Major             | Flagged             | Pass               | 12               |


## Step 2: Define Value Ranges and Probabilities
This a crucial step where you specify the range of values for each attribute and their probabilities to reflect realistic distributions.
1. Number of Clients Served Annually:
   - Less than 200 (40%): Smaller entities are often more numerous but have fewer clients. Assigning a higher probability here reflects this common scenario.
   - 201-500 (30%): Medium-sized entities are less common than smaller ones but still significant. A moderate probability reflects their prevalence.
   - Greater than 500 (30%): Large entities are fewer but can have substantial client bases. This probability balances the need to include enough large entities for meaningful analysis.
2. Past Infraction History Type:
   - None (50%): A significant portion of entities typically comply with regulations, so a higher probability is assigned.
   - Minor (30%): Minor infractions are common, but less so than having no infractions, thus a moderate probability.
   - Major (20%): Major infractions are less common, thus a lower probability is assigned.
3. Past Infraction History Timeline:
   - None (50%): Many entities may have a clean slate, so a higher probability is assigned.
   - Within past year (25%): Recent infractions are significant but less common than having none.
   - Within past 1-3 years (25%): Older infractions are equally likely as recent ones, providing a balanced distribution.
4. Public Complaints in Last Quarter:
   - None (60%): Most entities might not receive complaints frequently, reflecting a higher probability for no complaints.
   - Minor (25%): Minor complaints are less frequent than no complaints, hence a moderate probability.
   - Major (15%): Major complaints are relatively rare, reflecting the lowest probability.
5. Quarterly Public Sentiment Analysis:
   - None (70%): The majority of entities may not have flagged sentiments, reflecting a higher probability.
   - Flagged (30%): Negative sentiments are less common but significant for analysis, hence a lower probability.
6. Previous Inspection Results:
   - Pass (50%): Many entities might pass inspections, thus a higher probability is assigned.
   - Fail (25%): Failures are less common than passes, reflecting a moderate probability.
   - None (25%): Entities that haven't been inspected provide important data points, hence a moderate probability.
     
## Step 3: Create Synthetic Data Using Python
Refer the notebook: [SyntheticData.ipynb](Data/SyntheticData.ipynb)

## Step 4: Verify the Synthetic Data
- Review Distributions: Ensure the distributions of values match the expected probabilities.
- Check Risk Scores: Verify that the risk scores are calculated correctly based on the defined rules.
- Sample Validation:Manually inspect a few samples to ensure data integrity.
