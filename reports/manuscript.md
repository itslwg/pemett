```python
import os
os.chdir("..")
```

# EMETT: Ensemble Model for Emergency department Trauma Triage

## Introduction

Trauma is a major threat to population health globally [@Brohi2017; @GBD2017]. Every year about 4.6 million people die because of trauma - more than the deaths from HIV/AIDS, malaria and tuberculosis combined. This situation calls for not only more interventions, but also strengthened research on effective trauma care delivery.

Trauma care is highly time sensitive and delays to treatment have been associated with increased mortality across settings [@Yeboah2014; @OReilly2013; @Roy2017a]. Early identification and management of potentially life threatening injuries are crucial. Trauma triage - the process of prioritizing patients to match level of care with clinical acuity - is a key component of trauma care [@EAST2010; @NICE2016].

In health systems with formalised criteria for emergency department trauma triage, all patients are assigned a priority coupled with a target time to treat. These priorities are may be coded with numbers [@ESI2012] or colors [@SATG2012], for example red, orange, yellow and green, with red being assigned to the most urgent patients and green to the least urgent.

In health systems without formalized criteria, for example in many low resource settings, clinician gestalt is used informally triage trauma patients in the emergency department [@Baker2013]. Where prehospital care is lacking patients often arrive to the emergency department without warning [@Choi2017]. Identifying ways to quickly triage patients would therefore be valuable such settings.

The approach to triage trauma patients arriving to the emergency department has received little attention from the research community. Framed as a classification problem this challenge can be addressed using a statistical learner. Logistic or proportional hazards models are common classification learners whereas more modern alternatives include random forests or neural networks.

The uptake and use of such learners in trauma research has been slow [@Liu2017]. One recent study used a random forest learner to assign priority to patients in a general emergency department population, and found a slight performance improvement using this learner compared with the standard criteria [@Levin2018].

Given the paucity of research leveraging machine learning to triage trauma patients in the emergency department, we aimed to compare the performance of an ensemble machine learning model to that of clinician gestalt based on patients' presentation. Our hypothesis was that the performance of the this ensemble model would be non-inferior to that of clinician gestalt.

## Materials and Methods

### Study Design

We used data from the ongoing Trauma Triage Study (TTRIS) in India, a Towards Improved Trauma Care Outcomes (TITCO) study. This study is a prospective cohort study in three public hospitals in urban India.

### Study Setting

Data analysed for this study came from patients enrolled between 28 July 2016 and 21 November 2017 at the three hospitals Khershedji Behramji Bhabha hospital (KBBH) in Mumbai, Lok Nayak Hospital of
Maulana Azad Medical College (MAMC) in Delhi, and the Institute of Post-Graduate Medical Education and Research and Seth Sukhlal Karnani Memorial Hospital (SSKM) in Kolkata. The time frame was decided to ensure that all included patients had completed six months follow up. 

KBBH is a community hospital with 436 inpatient beds. There are departments of surgery, orthopaedics, anaesthesia, and both adult and paediatric intensive care units. It has a general ED where all
patients are seen.  Most patients present directly and are not transferred from another health centre. Plain X-rays and ultrasonography are available around the clock but computed tomography (CT) is only available in-house during day-time. During evenings and nights patients in need of a CT are referred elsewhere. 

MAMC and SSKM are both university and tertiary referral hospitals. This means that all specialities and imaging facilities relevant to trauma care, except emergency medicine, are available in-house around the clock. MAMC has approximately 2200 inpatient beds and SSKM has around 1775 inpatient beds. Both MAMC and SSKM have general emergency departments. Because both MAMC and SSKM are tertiary referral hospitals a large proportion of patients arriving at their EDs are transferred from other health facilities, with almost no transfer protocols in place.

Prehospital care is rudimentary in all three cities, with no organised emergency medical services. Ambulances are predominately used for inter-hospital transfers and most patients who arrive directly from the scene of the incident are brought by the police or in private vehicles. 

Patients arriving to the emergency department are at all centres first seen by a casualty medical officer on a largely first come first served basis. There is no formalised system for prioritising emergency department patients at any of the centres.

The research was approved by the ethical review board at each participating hospital. The names of the boards and the approval numbers were Ethics and Scientific Committee (KBBH, HO/4982/KBB), the Institutional Ethics Committee (MAMC, F.1/IEC/MAMC/53/2/2016/No97), and the IPGME&R Research Oversight Committee (SSKM,

### Data Collection

Data were collected by one dedicated project officer at each site. The project officers all had a masters degree in life sciences. They worked five shifts per week, and each shift was about eight hours long, so that mornings, evenings and nights were covered according to a rotating schedule. In each shift, project officers spent approximately six hours collecting data in the emergency department and the remaining two following up patients. The collected data were then transferred to a digital database. The rationale for this setup was to ensure collection of high-quality data from a representative sample of trauma patients arriving to the emergency departements at participating centres, while keeping to the projects budget constraints.

### Participants 

#### Eligibility criteria

Any person aged $\geq$ 18 years or older and who presented alive to the emergency department of participating sites with history of trauma was included. The age cutoff was chosen to align with Indian laws on research ethics and informed consent. We defined history of trauma as having any of the external causes of morbidity and mortality listed in block V01-Y36, chapter XX of the International Classification of Disease version 10 (ICD-10) code book as primary complaint. Drownings, inhalation and ingestion of objects causing obstruction of respiratory tract, contact with venomous snakes and lizards, accidental poisoning by and exposure to drugs, and overexertion were excluded because they are not considered trauma at the participating centres.

#### Source and methods of selection of participants and follow up

The project officers enrolled the first ten consecutive patients who presented to the emergency department during each shift. The number of patients to enrol was set to ten to make follow up feasible. Written informed consent from the patient or a patient representative was obtained either in the emergency department or in the ward if the patient was admitted. A follow-up was completed by the project officer 30 days and 6 months after participant arrived at participating hospital. The follow-up was completed in person or on phone, depending on whether the patient was still hospitalised or if the patient had been discharged. Phone numbers of one or more contact persons (e.g. relatives), were collected on enrolment and contacted if the participant did not reply on follow up. Only if neither the participant nor the contact person answered any of three repeated phone calls was the outcome recorded as missing and the patient was considered lost to follow up.

### Variables, Data Sources and Measurement 

#### Patient characteristics and ensemble model variables 

The ensemble model were trained on two target variables.  The first target was all-cause 30 day mortality, defined as death from any cause within 30 days of arrival to a participating centre. These data were extracted from patient records if the patient was still in hospital 30 days after arrival, or collected by calling the patient or the patient representative if the patient was not in hospital. The second target was a composite outcome of early mortality, ICU admission, major urgent surgery and severe injury. Early mortality was defined as death within 24 hours of arrival to  the participating hospitals. ICU admission was defined as admission to the ICU within 48 hours of arrival. While most ICU admissions occur within hours of hospital arrival, we extended the time frame to compensate for bed availability and transfer delays.  Major urgent surgery was defined as a major surgery performed within  24 hours of arrival to the participating hospitals. Surgical excerpts  were reworked into standardized nomenclature using the Nordic Medico-Statistical Committee (NOMESCO) Classification of surgical Procedures, and the Systematized Nomenclature in Medicine Clinical Terms (SNOMED CT). In the lack of a general definition of major surgery, a team of experienced surgeons and researches decided on what surgeries  to consider as major. Severe injury was defined as an Injury Severity Score (ISS) over 15. This cutoff-value for severe injury is traditionally sued in trauma research since it is said to indicate a 10 % mortality). **TAKEN FROM CELINAS PAPER**

The features included patient age in years, sex, mechanism of injury, type of injury, mode of transport, transfer status, time from injury to arrival in hours. The project officers collected data on these features by asking the patient, a patient representative, or by extracting the data from the patient's file. Sex was coded as male or female. Mechanism of injury was coded by the project officers using ICD-10 after completing the World Health Organization's (WHO) electronic ICD-10-training tool [@WHOICD]. The levels of mechanism of injury was collapsed for analysis into transport accident (codes V00-V99), falls (W00-W19), burns (X00-X19), intentional self harm (X60-X84), assault (X85-X99 and Y00-Y09), and other mechanism (W20-99, X20-59 and Y10-36). Type of injury was coded as blunt, penetrating, or both blunt and penetrating. Mode of transport was coded as ambulance, police, private vehicle, or arrived walking. Transfer status was a binary feature indicating if the patient was transferred from another health facility or not. 

The features also included vital signs measured on arrival to the ED at participating centres. The project officers recorded all vital signs using hand held equipment, i.e. these were not extracted from patient records, after receiving two days of training and yearly refreshers.  Only if the hand held equipment failed to record a value did the project officers extract data from other attached monitoring equipment, if available. Systolic and diastolic blood pressure (SBP and DBP) were measured using an automatic blood pressure monitor. Heart rate (HR) and peripheral capillary oxygen saturation (SpO~2~) were measured using a portable non-invasive fingertip pulse oximeter. Respiratory rate (RR) was measured manually by counting the number of breaths during one minute. Level of consciousness was measured using both the Glasgow coma scale (GCS) and the Alert, Voice, Pain, and Unresponsive scale (AVPU).  In assigning GCS the project officers used the official Glasgow Coma Scale Assessment Aid [@GCSAID]. AVPU simply indicates whether the patient is alert, responds to voice stimuli, painful stimuli, or does not respond at all. 

These represent standard variables commonly collected in many health systems. They are also included in several well known clinical prediction models designed to predict trauma mortality [@Rehn2011].

### Clinicians' priority levels
For the purpose of this study, clinicians were instructed by the project officers to assign a priority to each patient. The priority levels were color coded. Red was assigned to the most serious patients that should be treated first. Green was assigned to the least serious patients that should be treated last. Orange and yellow were intermediate levels, where orange patients were less serious than red but more serious than yellow and green whereas yellow patients were less serious than red and orange patients but more serious than green patients. The clinicians were allowed to use all information available at the time when they assigned the priority level, which was as soon as they had first seen the patient. The priorities were not used to guide further patient care and no interventions were implemented as part of the study for patients assigned to the more urgent priority levels.

### Bias
Project officers underwent two days of training in study procedures and were then supervised locally. We conducted continuous data quality assurance by having weekly online data review meetings during which data discrepancies were identified, discussed and resolved. We conducted quarterly on site quality control sessions during which data collection was conducted both by the centre's own project officer and a quality control officer. Data entry errors were prevented by having extensive logical checks in the digital data collection instrument.

### Statistical Methods
All data was de-identified before it was analysed for this study. Details of the de-identification procedures are available as supporting information. We used Python and R for all analysis [@R; @python]. In particular, we utilised packages [@TableOnePython] for generating sample characteristics table. We first made a non-random temporal split of the complete data set into a training and test set. The split was made so that 75% of the complete cohort was assigned to the training set and the remaining 25% to the test set, ensuring that the relative contribution of each centre was maintained in both sets. We then calculated descriptive statistics of all variables, using medians and inter quartile ranges (IQR) for continuous variables and counts and percentages for qualitative variables. All quantitative features (age, SBP, DBP, HR, SpO~2~, and RR) were treated as continuous and the levels of all qualitative variables (sex, mechanism of injury, type of injury, mode of transport, transfer status, and GCS components) were treated as bins (dummy variables).

#### Development of the Ensemble Model
The study sample was split into three parts, henceforth referred to as the training, validation, and test sets. We then developed our ensemble model in the training and validation setsusing the SuperLearner R package [@SuperLearner]. SuperLearner is an ensemblemachine learning algorithm, meaning that it combines predictions several learners to come up with an "optimal" learner. Table\@ref(tab:superlearner-library) shows our library of learners. Allwere implemented using the default hyperparameters. Short descriptionsof the individual learners are available as supportinginformation. 

The ensemble model was trained using ten fold cross validation. This procedure is implemented by default in the SuperLearner package and entails splitting the development data in ten mutually exclusive parts of approximately the same size. All learners included in the library are then fitted using the combined data of nine of these parts and evaluated in the tenth. This procedure is then repeated ten times, i.e. each part is used once as the evaluation data, and is intended to limit overfitting and reduce optimism.

The ensemble model predictions was then used to assign levels of priority  to patients. This was done by binning the ensemble model prediction into four bins using cutoffs identified using a grid search to optimize the area under the receiver operation characteristics curve (AUROCC) across all possible combinations of unique cutoffs, where each cutoff could take any value from 0.01 to 0.99 in 0.01 unit increments. These bins corresponded to the green, yellow, orange, and red priority levels assigned by the clinicians. The cutoffs were identified in the validation set in order  to prevent information leakage and limit bias. The performance of both the continuous ensemble model prediction and the ensemble model priority levels was then evaluated by estimating their AUROCC. We also visualised the performance by plotting ROC curves.

| Algorithm                 | Package      |
|---------------------------|--------------|
| Gradient Boosting Machine | LightGBM     |
| Random Forest Classifier  | scikit-learn |
| Multi-layer Perceptron    | pytorch      |

#### Comparing the ensemble model and Clinicians

We then used the ensemble model to predict the outcomes of the patients in the test set and used the cutoff values from the validation set to assign a level of priority to each patient in this set. The performance of the continuous ensemble prediction, the ensemble model priority levels, and the clinicians' priority levels, was then evaluated by estimating and comparing their AUROCC. 

The levels of priority assigned by the ensemble model and clinicians respectively were then compared by estimating the net reclassification, in events (patient with the outcome, i.e. who died within 30-days from arrival) and non-events (patient without the outcome) respectively. The net reclassification in events was defined as the difference between the proportion of events assigned a higher priority by the ensemble model than the clinicians and the proportion of events assigned a lower priority by the SuperLearner than the clinicians. Conversely, the net reclassification in non-events was defined as the difference between the proportion of non-events assigned to a lower priority by the ensemble model than the clinicians and the proportion of non-events assigned a higher priority by the SuperLearner than the clinicians. 

We used an empirical bootstrap with 1000 draws of the same size as the original set to estimate 95% confidence interval (CI) around differences. We concluded that the SuperLearner was non-inferior to clinicians if the 95% CI of the net reclassification in events did not exceed a pre-specified level of -0.05, indicating that clinicians correctly classified 5 in 100 events more than the ensemble model.

#### Handling of Missing Data
Observations with missing data on all cause 30-day mortality or priority level assigned by clinicians were excluded. Missing data in features was treated as informative. For each feature with missing data we created a non-missingness indicator, a variable that took the value of 0 if the feature value was missing and 1 otherwise. Missing feature values were then replaced with the median of observed data for quantitative features and the most common level for qualitative features. We included the non-missingness indicators as features in the ensemble model.


## Results
During the study period, we approached a total of xxx patients for enrolment.
A random sample of 
`r round(as.integer(results$s30d.descriptive.statistics$n.enrolled)*0.01)` 
observations were removed during data de-identification. Consent was declined by `r results$s30d.descriptive.statistics$n.not.consent` patients. Out of the 
`r results$s30d.descriptive.statistics$n.consent` patients who provided informed consent,
`r results$s30d.descriptive.statistics$n.not.adults` were not adults, and
`r results$s30d.descriptive.statistics$n.not.has.triage.category` had missing information on
triage category, leaving `r results$s30d.descriptive.statistics$n.has.triage.category`.
For 30-day mortality, 
`r results$s30d.descriptive.statistics$n.not.complete.outcome` had missing outcome data, whereas for the composite outcome `r results$composite.descriptive.statistics$n.not.complete.outcome` had missing data. 
Thus, the final samples included `r results$s30d.descriptive.statistics$n.complete.outcome` and
`r results$composite.descriptive.statistics$n.complete.outcome` patients for the 30-day mortality and composite outcomes, respectively.


```python
import pandas as pd

from src.visualization.visualize import create_sample_characteristics_table

data_dir = data_dir = "./data/"
data_dictionary = pd.read_csv(
    data_dir + "raw/data_dictionary.csv",
    delimiter = ","
)
cat_features = ['moi', 'sex', 'mot', 'tran', 'egcs', 'mgcs', 'vgcs', 'avpu']
cont_features = ['age', 'hr', 'sbp', 'dbp', 'spo2', 'rr', 'delay']
```


```python
# Read relevant files
df = pd.read_csv(data_dir + "interim/table_sample_s30d.csv")
data_dictionary = pd.read_csv(
    data_dir + "raw/data_dictionary.csv",
    delimiter = ","
)
create_sample_characteristics_table(
    df=df,
    data_dictionary=data_dictionary,
    categorical=cat_features + ["s30d", "tc"],
    nonnormal=cont_features, 
    groupby="partition"
)
```

    /Users/ludvigwarnberggerdin/miniforge3/envs/pemett/lib/python3.10/site-packages/tableone/tableone.py:991: FutureWarning: Using the level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().
    /Users/ludvigwarnberggerdin/miniforge3/envs/pemett/lib/python3.10/site-packages/tableone/tableone.py:991: FutureWarning: Using the level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="4" halign="left">Grouped by partition</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Missing</th>
      <th>Overall</th>
      <th>Holdout</th>
      <th>Train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>n</th>
      <th></th>
      <td></td>
      <td>8295</td>
      <td>2074</td>
      <td>6221</td>
    </tr>
    <tr>
      <th>Unnamed: 0, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>5629.4 (3293.1)</td>
      <td>5627.4 (3334.0)</td>
      <td>5630.0 (3279.6)</td>
    </tr>
    <tr>
      <th>Age in years, median [Q1,Q3]</th>
      <th></th>
      <td>0</td>
      <td>32.0 [24.0,45.0]</td>
      <td>31.0 [24.0,45.0]</td>
      <td>32.0 [24.0,45.0]</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Mechanism of injury, n (%)</th>
      <th>Assault</th>
      <td>0</td>
      <td>1191 (14.4)</td>
      <td>285 (13.7)</td>
      <td>906 (14.6)</td>
    </tr>
    <tr>
      <th>Burns</th>
      <td></td>
      <td>48 (0.6)</td>
      <td>14 (0.7)</td>
      <td>34 (0.5)</td>
    </tr>
    <tr>
      <th>Event of undetermined intent</th>
      <td></td>
      <td>4 (0.0)</td>
      <td></td>
      <td>4 (0.1)</td>
    </tr>
    <tr>
      <th>Falls</th>
      <td></td>
      <td>2309 (27.8)</td>
      <td>600 (28.9)</td>
      <td>1709 (27.5)</td>
    </tr>
    <tr>
      <th>Intentional self-harm</th>
      <td></td>
      <td>36 (0.4)</td>
      <td>10 (0.5)</td>
      <td>26 (0.4)</td>
    </tr>
    <tr>
      <th>Other external causes of accidental injury</th>
      <td></td>
      <td>1423 (17.2)</td>
      <td>334 (16.1)</td>
      <td>1089 (17.5)</td>
    </tr>
    <tr>
      <th>Transportation accident</th>
      <td></td>
      <td>651 (7.8)</td>
      <td>147 (7.1)</td>
      <td>504 (8.1)</td>
    </tr>
    <tr>
      <th>Unlabelled</th>
      <td></td>
      <td>2633 (31.7)</td>
      <td>684 (33.0)</td>
      <td>1949 (31.3)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sex, n (%)</th>
      <th>Female</th>
      <td>1</td>
      <td>1885 (22.7)</td>
      <td>445 (21.5)</td>
      <td>1440 (23.2)</td>
    </tr>
    <tr>
      <th>Male</th>
      <td></td>
      <td>6409 (77.3)</td>
      <td>1629 (78.5)</td>
      <td>4780 (76.8)</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Mode of transport, n (%)</th>
      <th>Ambulance</th>
      <td>4</td>
      <td>3083 (37.2)</td>
      <td>790 (38.1)</td>
      <td>2293 (36.9)</td>
    </tr>
    <tr>
      <th>Arrived walking</th>
      <td></td>
      <td>371 (4.5)</td>
      <td>88 (4.2)</td>
      <td>283 (4.6)</td>
    </tr>
    <tr>
      <th>Police</th>
      <td></td>
      <td>184 (2.2)</td>
      <td>43 (2.1)</td>
      <td>141 (2.3)</td>
    </tr>
    <tr>
      <th>Private vehicle</th>
      <td></td>
      <td>4653 (56.1)</td>
      <td>1152 (55.6)</td>
      <td>3501 (56.3)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Transferred from another health facility, n (%)</th>
      <th>No</th>
      <td>0</td>
      <td>4691 (56.6)</td>
      <td>1160 (55.9)</td>
      <td>3531 (56.8)</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td></td>
      <td>3604 (43.4)</td>
      <td>914 (44.1)</td>
      <td>2690 (43.2)</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Eye component of the Glasgow Coma Scale, n (%)</th>
      <th>1</th>
      <td>1</td>
      <td>289 (3.5)</td>
      <td>68 (3.3)</td>
      <td>221 (3.6)</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>149 (1.8)</td>
      <td>44 (2.1)</td>
      <td>105 (1.7)</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>202 (2.4)</td>
      <td>50 (2.4)</td>
      <td>152 (2.4)</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>7615 (91.8)</td>
      <td>1902 (91.7)</td>
      <td>5713 (91.8)</td>
    </tr>
    <tr>
      <th>Non testable</th>
      <td></td>
      <td>39 (0.5)</td>
      <td>10 (0.5)</td>
      <td>29 (0.5)</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Motor component of the Glasgow Coma Scale, n (%)</th>
      <th>1</th>
      <td>1</td>
      <td>98 (1.2)</td>
      <td>12 (0.6)</td>
      <td>86 (1.4)</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>63 (0.8)</td>
      <td>17 (0.8)</td>
      <td>46 (0.7)</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>59 (0.7)</td>
      <td>16 (0.8)</td>
      <td>43 (0.7)</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>86 (1.0)</td>
      <td>21 (1.0)</td>
      <td>65 (1.0)</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>364 (4.4)</td>
      <td>100 (4.8)</td>
      <td>264 (4.2)</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>7620 (91.9)</td>
      <td>1906 (91.9)</td>
      <td>5714 (91.9)</td>
    </tr>
    <tr>
      <th>Non testable</th>
      <td></td>
      <td>4 (0.0)</td>
      <td>1 (0.0)</td>
      <td>3 (0.0)</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Verbal component of the Glasgow Coma Scale, n (%)</th>
      <th>1</th>
      <td>0</td>
      <td>303 (3.7)</td>
      <td>71 (3.4)</td>
      <td>232 (3.7)</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>172 (2.1)</td>
      <td>48 (2.3)</td>
      <td>124 (2.0)</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>96 (1.2)</td>
      <td>22 (1.1)</td>
      <td>74 (1.2)</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>365 (4.4)</td>
      <td>96 (4.6)</td>
      <td>269 (4.3)</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>7348 (88.6)</td>
      <td>1834 (88.4)</td>
      <td>5514 (88.6)</td>
    </tr>
    <tr>
      <th>Non testable</th>
      <td></td>
      <td>11 (0.1)</td>
      <td>3 (0.1)</td>
      <td>8 (0.1)</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Alert, voice, pain, unresponsive scale, n (%)</th>
      <th>Alert</th>
      <td>1</td>
      <td>7569 (91.3)</td>
      <td>1887 (91.0)</td>
      <td>5682 (91.3)</td>
    </tr>
    <tr>
      <th>Pain responsive</th>
      <td></td>
      <td>477 (5.8)</td>
      <td>128 (6.2)</td>
      <td>349 (5.6)</td>
    </tr>
    <tr>
      <th>Unresponsive</th>
      <td></td>
      <td>93 (1.1)</td>
      <td>15 (0.7)</td>
      <td>78 (1.3)</td>
    </tr>
    <tr>
      <th>Voice responsive</th>
      <td></td>
      <td>155 (1.9)</td>
      <td>43 (2.1)</td>
      <td>112 (1.8)</td>
    </tr>
    <tr>
      <th>Heart rate, median [Q1,Q3]</th>
      <th></th>
      <td>3</td>
      <td>83.0 [76.0,93.0]</td>
      <td>83.0 [76.0,93.0]</td>
      <td>83.0 [76.0,93.0]</td>
    </tr>
    <tr>
      <th>Systolic blood pressure in mmHg, median [Q1,Q3]</th>
      <th></th>
      <td>18</td>
      <td>123.0 [112.0,132.0]</td>
      <td>124.0 [113.0,132.0]</td>
      <td>123.0 [112.0,132.0]</td>
    </tr>
    <tr>
      <th>Diastolic blood pressure in mmHg, median [Q1,Q3]</th>
      <th></th>
      <td>19</td>
      <td>80.0 [72.0,88.0]</td>
      <td>80.0 [72.0,88.0]</td>
      <td>80.0 [72.0,88.0]</td>
    </tr>
    <tr>
      <th>Peripheral capillary oxygen saturation, median [Q1,Q3]</th>
      <th></th>
      <td>2</td>
      <td>98.0 [98.0,98.0]</td>
      <td>98.0 [97.0,98.0]</td>
      <td>98.0 [98.0,98.0]</td>
    </tr>
    <tr>
      <th>Respiratory rate in breaths per minute, median [Q1,Q3]</th>
      <th></th>
      <td>6</td>
      <td>22.0 [20.0,24.0]</td>
      <td>22.0 [20.0,24.0]</td>
      <td>22.0 [20.0,24.0]</td>
    </tr>
    <tr>
      <th>Time between injury and arrival to participating centre in minutes, median [Q1,Q3]</th>
      <th></th>
      <td>40</td>
      <td>0.0 [0.0,0.0]</td>
      <td>0.0 [0.0,0.0]</td>
      <td>0.0 [0.0,0.0]</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">All cause 30-day mortality, n (%)</th>
      <th>No</th>
      <td>0</td>
      <td>7814 (94.2)</td>
      <td>1954 (94.2)</td>
      <td>5860 (94.2)</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td></td>
      <td>481 (5.8)</td>
      <td>120 (5.8)</td>
      <td>361 (5.8)</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Level of priority assigned by clinicians, n (%)</th>
      <th>Green</th>
      <td>0</td>
      <td>4602 (55.5)</td>
      <td>1139 (54.9)</td>
      <td>3463 (55.7)</td>
    </tr>
    <tr>
      <th>Orange</th>
      <td></td>
      <td>650 (7.8)</td>
      <td>170 (8.2)</td>
      <td>480 (7.7)</td>
    </tr>
    <tr>
      <th>Red</th>
      <td></td>
      <td>373 (4.5)</td>
      <td>83 (4.0)</td>
      <td>290 (4.7)</td>
    </tr>
    <tr>
      <th>Yellow</th>
      <td></td>
      <td>2670 (32.2)</td>
      <td>682 (32.9)</td>
      <td>1988 (32.0)</td>
    </tr>
  </tbody>
</table>
</div><br />




```python
# Read relevant files
df = pd.read_csv(data_dir + "interim/table_sample_composite.csv")
data_dictionary = pd.read_csv(
    data_dir + "raw/data_dictionary.csv",
    delimiter = ","
)
t1 = create_sample_characteristics_table(
    df=df,
    data_dictionary=data_dictionary,
    categorical=cat_features + ["composite", "tc"],
    nonnormal=cont_features, 
    groupby="partition"
)
```

    /Users/ludvigwarnberggerdin/miniforge3/envs/pemett/lib/python3.10/site-packages/tableone/tableone.py:991: FutureWarning: Using the level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().



```python
print(t1.tabulate())
```

    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    |                                            | Missing   | Overall             | Holdout             | Train               |
    +====================================================================================+============================================+===========+=====================+=====================+=====================+
    | n                                                                                  |                                            |           | 8295                | 2074                | 6221                |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Unnamed: 0, mean (SD)                                                              |                                            | 0         | 5629.4 (3293.1)     | 5597.2 (3278.4)     | 5640.1 (3298.2)     |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Age in years, median [Q1,Q3]                                                       |                                            | 0         | 32.0 [24.0,45.0]    | 32.0 [24.0,45.0]    | 32.0 [24.0,45.0]    |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Mechanism of injury, n (%)                                                         | Assault                                    | 0         | 1191 (14.4)         | 297 (14.3)          | 894 (14.4)          |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Burns                                      |           | 48 (0.6)            | 12 (0.6)            | 36 (0.6)            |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Event of undetermined intent               |           | 4 (0.0)             |                     | 4 (0.1)             |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Falls                                      |           | 2309 (27.8)         | 558 (26.9)          | 1751 (28.1)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Intentional self-harm                      |           | 36 (0.4)            | 8 (0.4)             | 28 (0.5)            |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Other external causes of accidental injury |           | 1423 (17.2)         | 351 (16.9)          | 1072 (17.2)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Transportation accident                    |           | 651 (7.8)           | 158 (7.6)           | 493 (7.9)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Unlabelled                                 |           | 2633 (31.7)         | 690 (33.3)          | 1943 (31.2)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Sex, n (%)                                                                         | Female                                     | 1         | 1885 (22.7)         | 460 (22.2)          | 1425 (22.9)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Male                                       |           | 6409 (77.3)         | 1613 (77.8)         | 4796 (77.1)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Mode of transport, n (%)                                                           | Ambulance                                  | 4         | 3083 (37.2)         | 778 (37.5)          | 2305 (37.1)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Arrived walking                            |           | 371 (4.5)           | 87 (4.2)            | 284 (4.6)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Police                                     |           | 184 (2.2)           | 49 (2.4)            | 135 (2.2)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Private vehicle                            |           | 4653 (56.1)         | 1159 (55.9)         | 3494 (56.2)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Transferred from another health facility, n (%)                                    | No                                         | 0         | 4691 (56.6)         | 1174 (56.6)         | 3517 (56.5)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Yes                                        |           | 3604 (43.4)         | 900 (43.4)          | 2704 (43.5)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Eye component of the Glasgow Coma Scale, n (%)                                     | 1                                          | 1         | 289 (3.5)           | 71 (3.4)            | 218 (3.5)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 2                                          |           | 149 (1.8)           | 40 (1.9)            | 109 (1.8)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 3                                          |           | 202 (2.4)           | 50 (2.4)            | 152 (2.4)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 4                                          |           | 7615 (91.8)         | 1900 (91.6)         | 5715 (91.9)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Non testable                               |           | 39 (0.5)            | 13 (0.6)            | 26 (0.4)            |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Motor component of the Glasgow Coma Scale, n (%)                                   | 1                                          | 1         | 98 (1.2)            | 20 (1.0)            | 78 (1.3)            |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 2                                          |           | 63 (0.8)            | 15 (0.7)            | 48 (0.8)            |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 3                                          |           | 59 (0.7)            | 14 (0.7)            | 45 (0.7)            |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 4                                          |           | 86 (1.0)            | 26 (1.3)            | 60 (1.0)            |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 5                                          |           | 364 (4.4)           | 95 (4.6)            | 269 (4.3)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 6                                          |           | 7620 (91.9)         | 1901 (91.7)         | 5719 (91.9)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Non testable                               |           | 4 (0.0)             | 2 (0.1)             | 2 (0.0)             |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Verbal component of the Glasgow Coma Scale, n (%)                                  | 1                                          | 0         | 303 (3.7)           | 65 (3.1)            | 238 (3.8)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 2                                          |           | 172 (2.1)           | 44 (2.1)            | 128 (2.1)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 3                                          |           | 96 (1.2)            | 29 (1.4)            | 67 (1.1)            |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 4                                          |           | 365 (4.4)           | 93 (4.5)            | 272 (4.4)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | 5                                          |           | 7348 (88.6)         | 1841 (88.8)         | 5507 (88.5)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Non testable                               |           | 11 (0.1)            | 2 (0.1)             | 9 (0.1)             |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Alert, voice, pain, unresponsive scale, n (%)                                      | Alert                                      | 1         | 7569 (91.3)         | 1890 (91.2)         | 5679 (91.3)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Pain responsive                            |           | 477 (5.8)           | 120 (5.8)           | 357 (5.7)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Unresponsive                               |           | 93 (1.1)            | 18 (0.9)            | 75 (1.2)            |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Voice responsive                           |           | 155 (1.9)           | 45 (2.2)            | 110 (1.8)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Heart rate, median [Q1,Q3]                                                         |                                            | 3         | 83.0 [76.0,93.0]    | 82.0 [76.0,92.0]    | 83.0 [76.0,94.0]    |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Systolic blood pressure in mmHg, median [Q1,Q3]                                    |                                            | 18        | 123.0 [112.0,132.0] | 124.0 [113.0,132.0] | 123.0 [112.0,132.0] |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Diastolic blood pressure in mmHg, median [Q1,Q3]                                   |                                            | 19        | 80.0 [72.0,88.0]    | 80.0 [72.0,88.0]    | 80.0 [72.0,88.0]    |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Peripheral capillary oxygen saturation, median [Q1,Q3]                             |                                            | 2         | 98.0 [98.0,98.0]    | 98.0 [98.0,98.0]    | 98.0 [97.8,98.0]    |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Respiratory rate in breaths per minute, median [Q1,Q3]                             |                                            | 6         | 22.0 [20.0,24.0]    | 22.0 [20.0,24.0]    | 22.0 [20.0,24.0]    |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Time between injury and arrival to participating centre in minutes, median [Q1,Q3] |                                            | 40        | 0.0 [0.0,0.0]       | 0.0 [0.0,0.0]       | 0.0 [0.0,0.0]       |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Composite outcome, n (%)                                                           | No                                         | 0         | 8143 (98.2)         | 2036 (98.2)         | 6107 (98.2)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Yes                                        |           | 152 (1.8)           | 38 (1.8)            | 114 (1.8)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    | Level of priority assigned by clinicians, n (%)                                    | Green                                      | 0         | 4602 (55.5)         | 1180 (56.9)         | 3422 (55.0)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Orange                                     |           | 650 (7.8)           | 156 (7.5)           | 494 (7.9)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Red                                        |           | 373 (4.5)           | 82 (4.0)            | 291 (4.7)           |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+
    |                                                                                    | Yellow                                     |           | 2670 (32.2)         | 656 (31.6)          | 2014 (32.4)         |
    +------------------------------------------------------------------------------------+--------------------------------------------+-----------+---------------------+---------------------+---------------------+



```python
!jupyter nbconvert --to markdown ./reports/manuscript.ipynb
```

    [NbConvertApp] Converting notebook ./reports/manuscript.ipynb to markdown
    [NbConvertApp] Writing 45848 bytes to reports/manuscript.md



```python
!pandoc -s ./reports/manuscript.md -t docx -o ./reports/manuscript.docx --citeproc --bibliography="./reports/bibliography.bib" --csl="./reports/apa.csl"

```
