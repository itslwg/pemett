```python
import os
os.chdir("..")
```


```python
%load_ext autoreload
%autoreload 2
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


```python
import pandas as pd
data_dir = "./data/"
excl = pd.read_csv(data_dir + "interim/excl.csv", index_col = 0).iloc[:, 0]
```


```python
excl
```




    original               12412
    n_removed_ic             533
    n_ic                   11879
    n_removed_age              7
    n_age                  11872
    n_removed_tc             297
    n_tc                   11575
    n_removed_s30d          3255
    n_s30d                  8320
    n_removed_composite        0
    n_composite            11575
    Name: 0, dtype: int64




```python
round(excl.original * 0.01)
```




    124



During the study period, we approached a total of xxx patients for enrolment.
A random sample of 124 observations were removed during data de-identification. Consent was declined by 533 patients. Out of the 11879 patients who provided informed consent,
7 were not adults, and 297 had missing information on
triage cateogory, leaving 11575 patients.
For 30-day mortality, 3280 patients had missing outcome data, whereas for the composite outcome 0 patients had missing data. Thus, the final samples included 8295 and
11575 patients for the 30-day mortality and composite outcomes, respectively.


```python
from src.visualization.manuscript import generate_sample_characteristics_table
```


```python
generate_sample_characteristics_table("s30d")
```

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
      <td>8320</td>
      <td>2080</td>
      <td>6240</td>
    </tr>
    <tr>
      <th>Age in years, median [Q1,Q3]</th>
      <th></th>
      <td>0</td>
      <td>32.0 [24.0,45.0]</td>
      <td>32.0 [24.0,45.0]</td>
      <td>32.0 [24.0,45.0]</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Mechanism of injury, n (%)</th>
      <th>0</th>
      <td>0</td>
      <td>1192 (14.3)</td>
      <td>287 (13.8)</td>
      <td>905 (14.5)</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>48 (0.6)</td>
      <td>14 (0.7)</td>
      <td>34 (0.5)</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>597 (7.2)</td>
      <td>593 (28.5)</td>
      <td>4 (0.1)</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>1732 (20.8)</td>
      <td>12 (0.6)</td>
      <td>1720 (27.6)</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>353 (4.2)</td>
      <td>329 (15.8)</td>
      <td>24 (0.4)</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>1247 (15.0)</td>
      <td>151 (7.3)</td>
      <td>1096 (17.6)</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>1203 (14.5)</td>
      <td>694 (33.4)</td>
      <td>509 (8.2)</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>1948 (23.4)</td>
      <td></td>
      <td>1948 (31.2)</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Sex, n (%)</th>
      <th>2</th>
      <td>0</td>
      <td>1 (0.0)</td>
      <td>1 (0.0)</td>
      <td></td>
    </tr>
    <tr>
      <th>Female</th>
      <td></td>
      <td>1889 (22.7)</td>
      <td>458 (22.0)</td>
      <td>1431 (22.9)</td>
    </tr>
    <tr>
      <th>Male</th>
      <td></td>
      <td>6430 (77.3)</td>
      <td>1621 (77.9)</td>
      <td>4809 (77.1)</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Mode of transport, n (%)</th>
      <th>4</th>
      <td>0</td>
      <td>4 (0.0)</td>
      <td>1 (0.0)</td>
      <td>3 (0.0)</td>
    </tr>
    <tr>
      <th>Ambulance</th>
      <td></td>
      <td>3094 (37.2)</td>
      <td>797 (38.3)</td>
      <td>2297 (36.8)</td>
    </tr>
    <tr>
      <th>Arrived walking</th>
      <td></td>
      <td>371 (4.5)</td>
      <td>83 (4.0)</td>
      <td>288 (4.6)</td>
    </tr>
    <tr>
      <th>Police</th>
      <td></td>
      <td>185 (2.2)</td>
      <td>51 (2.5)</td>
      <td>134 (2.1)</td>
    </tr>
    <tr>
      <th>Private vehicle</th>
      <td></td>
      <td>4666 (56.1)</td>
      <td>1148 (55.2)</td>
      <td>3518 (56.4)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Transferred from another health facility, n (%)</th>
      <th>No</th>
      <td>0</td>
      <td>4702 (56.5)</td>
      <td>1163 (55.9)</td>
      <td>3539 (56.7)</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td></td>
      <td>3618 (43.5)</td>
      <td>917 (44.1)</td>
      <td>2701 (43.3)</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Eye component of the Glasgow Coma Scale, n (%)</th>
      <th>0</th>
      <td>0</td>
      <td>306 (3.7)</td>
      <td>78 (3.8)</td>
      <td>228 (3.7)</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>151 (1.8)</td>
      <td>38 (1.8)</td>
      <td>113 (1.8)</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>202 (2.4)</td>
      <td>51 (2.5)</td>
      <td>151 (2.4)</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>7620 (91.6)</td>
      <td>1905 (91.6)</td>
      <td>5715 (91.6)</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>39 (0.5)</td>
      <td>8 (0.4)</td>
      <td>31 (0.5)</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>2 (0.0)</td>
      <td></td>
      <td>2 (0.0)</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Motor component of the Glasgow Coma Scale, n (%)</th>
      <th>0</th>
      <td>0</td>
      <td>112 (1.3)</td>
      <td>19 (0.9)</td>
      <td>93 (1.5)</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>64 (0.8)</td>
      <td>17 (0.8)</td>
      <td>47 (0.8)</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>61 (0.7)</td>
      <td>15 (0.7)</td>
      <td>46 (0.7)</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>87 (1.0)</td>
      <td>24 (1.2)</td>
      <td>63 (1.0)</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>366 (4.4)</td>
      <td>98 (4.7)</td>
      <td>268 (4.3)</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>7624 (91.6)</td>
      <td>1905 (91.6)</td>
      <td>5719 (91.7)</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>4 (0.0)</td>
      <td>1 (0.0)</td>
      <td>3 (0.0)</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>2 (0.0)</td>
      <td>1 (0.0)</td>
      <td>1 (0.0)</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Verbal component of the Glasgow Coma Scale, n (%)</th>
      <th>0</th>
      <td>0</td>
      <td>316 (3.8)</td>
      <td>77 (3.7)</td>
      <td>239 (3.8)</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>175 (2.1)</td>
      <td>50 (2.4)</td>
      <td>125 (2.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>100 (1.2)</td>
      <td>23 (1.1)</td>
      <td>77 (1.2)</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>366 (4.4)</td>
      <td>96 (4.6)</td>
      <td>270 (4.3)</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>7351 (88.4)</td>
      <td>1832 (88.1)</td>
      <td>5519 (88.4)</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>11 (0.1)</td>
      <td>2 (0.1)</td>
      <td>9 (0.1)</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>1 (0.0)</td>
      <td></td>
      <td>1 (0.0)</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Alert, voice, pain, unresponsive scale, n (%)</th>
      <th>4</th>
      <td>0</td>
      <td>2 (0.0)</td>
      <td>1 (0.0)</td>
      <td>1 (0.0)</td>
    </tr>
    <tr>
      <th>Alert</th>
      <td></td>
      <td>7575 (91.0)</td>
      <td>1886 (90.7)</td>
      <td>5689 (91.2)</td>
    </tr>
    <tr>
      <th>Pain responsive</th>
      <td></td>
      <td>484 (5.8)</td>
      <td>132 (6.3)</td>
      <td>352 (5.6)</td>
    </tr>
    <tr>
      <th>Unresponsive</th>
      <td></td>
      <td>103 (1.2)</td>
      <td>23 (1.1)</td>
      <td>80 (1.3)</td>
    </tr>
    <tr>
      <th>Voice responsive</th>
      <td></td>
      <td>156 (1.9)</td>
      <td>38 (1.8)</td>
      <td>118 (1.9)</td>
    </tr>
    <tr>
      <th>Heart rate, median [Q1,Q3]</th>
      <th></th>
      <td>10</td>
      <td>83.0 [76.0,93.0]</td>
      <td>82.0 [76.0,94.0]</td>
      <td>83.0 [76.0,93.0]</td>
    </tr>
    <tr>
      <th>Systolic blood pressure in mmHg, median [Q1,Q3]</th>
      <th></th>
      <td>25</td>
      <td>123.0 [112.0,132.0]</td>
      <td>124.0 [113.0,133.0]</td>
      <td>123.0 [112.0,132.0]</td>
    </tr>
    <tr>
      <th>Diastolic blood pressure in mmHg, median [Q1,Q3]</th>
      <th></th>
      <td>26</td>
      <td>80.0 [72.0,88.0]</td>
      <td>80.0 [72.0,88.0]</td>
      <td>80.0 [72.0,88.0]</td>
    </tr>
    <tr>
      <th>Peripheral capillary oxygen saturation, median [Q1,Q3]</th>
      <th></th>
      <td>8</td>
      <td>98.0 [98.0,98.0]</td>
      <td>98.0 [97.0,98.0]</td>
      <td>98.0 [98.0,98.0]</td>
    </tr>
    <tr>
      <th>Respiratory rate in breaths per minute, median [Q1,Q3]</th>
      <th></th>
      <td>13</td>
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
      <td>7814 (93.9)</td>
      <td>1953 (93.9)</td>
      <td>5861 (93.9)</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td></td>
      <td>506 (6.1)</td>
      <td>127 (6.1)</td>
      <td>379 (6.1)</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Level of priority assigned by clinicians, n (%)</th>
      <th>Green</th>
      <td>0</td>
      <td>4604 (55.3)</td>
      <td>1138 (54.7)</td>
      <td>3466 (55.5)</td>
    </tr>
    <tr>
      <th>Orange</th>
      <td></td>
      <td>652 (7.8)</td>
      <td>185 (8.9)</td>
      <td>467 (7.5)</td>
    </tr>
    <tr>
      <th>Red</th>
      <td></td>
      <td>392 (4.7)</td>
      <td>79 (3.8)</td>
      <td>313 (5.0)</td>
    </tr>
    <tr>
      <th>Yellow</th>
      <td></td>
      <td>2672 (32.1)</td>
      <td>678 (32.6)</td>
      <td>1994 (32.0)</td>
    </tr>
  </tbody>
</table>
</div><br />




```python
generate_sample_characteristics_table("composite")
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
      <td>11575</td>
      <td>2894</td>
      <td>8681</td>
    </tr>
    <tr>
      <th>Age in years, median [Q1,Q3]</th>
      <th></th>
      <td>0</td>
      <td>32.0 [24.0,45.0]</td>
      <td>32.0 [24.0,45.0]</td>
      <td>32.0 [24.0,45.0]</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Mechanism of injury, n (%)</th>
      <th>Assault</th>
      <td>0</td>
      <td>1726 (14.9)</td>
      <td>447 (15.4)</td>
      <td>1279 (14.7)</td>
    </tr>
    <tr>
      <th>Burns</th>
      <td></td>
      <td>76 (0.7)</td>
      <td>11 (0.4)</td>
      <td>65 (0.7)</td>
    </tr>
    <tr>
      <th>Event of undetermined intent</th>
      <td></td>
      <td>12 (0.1)</td>
      <td>3 (0.1)</td>
      <td>9 (0.1)</td>
    </tr>
    <tr>
      <th>Falls</th>
      <td></td>
      <td>3154 (27.2)</td>
      <td>734 (25.4)</td>
      <td>2420 (27.9)</td>
    </tr>
    <tr>
      <th>Intentional self-harm</th>
      <td></td>
      <td>47 (0.4)</td>
      <td>12 (0.4)</td>
      <td>35 (0.4)</td>
    </tr>
    <tr>
      <th>Other external causes of accidental injury</th>
      <td></td>
      <td>2131 (18.4)</td>
      <td>561 (19.4)</td>
      <td>1570 (18.1)</td>
    </tr>
    <tr>
      <th>Transportation accident</th>
      <td></td>
      <td>981 (8.5)</td>
      <td>220 (7.6)</td>
      <td>761 (8.8)</td>
    </tr>
    <tr>
      <th>Unlabelled</th>
      <td></td>
      <td>3448 (29.8)</td>
      <td>906 (31.3)</td>
      <td>2542 (29.3)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sex, n (%)</th>
      <th>Female</th>
      <td>6</td>
      <td>2638 (22.8)</td>
      <td>649 (22.4)</td>
      <td>1989 (22.9)</td>
    </tr>
    <tr>
      <th>Male</th>
      <td></td>
      <td>8931 (77.2)</td>
      <td>2243 (77.6)</td>
      <td>6688 (77.1)</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Mode of transport, n (%)</th>
      <th>Ambulance</th>
      <td>6</td>
      <td>3782 (32.7)</td>
      <td>939 (32.5)</td>
      <td>2843 (32.8)</td>
    </tr>
    <tr>
      <th>Arrived walking</th>
      <td></td>
      <td>510 (4.4)</td>
      <td>123 (4.3)</td>
      <td>387 (4.5)</td>
    </tr>
    <tr>
      <th>Police</th>
      <td></td>
      <td>278 (2.4)</td>
      <td>78 (2.7)</td>
      <td>200 (2.3)</td>
    </tr>
    <tr>
      <th>Private vehicle</th>
      <td></td>
      <td>6999 (60.5)</td>
      <td>1753 (60.6)</td>
      <td>5246 (60.5)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Transferred from another health facility, n (%)</th>
      <th>No</th>
      <td>1</td>
      <td>7042 (60.8)</td>
      <td>1779 (61.5)</td>
      <td>5263 (60.6)</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td></td>
      <td>4532 (39.2)</td>
      <td>1115 (38.5)</td>
      <td>3417 (39.4)</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Eye component of the Glasgow Coma Scale, n (%)</th>
      <th>1</th>
      <td>4</td>
      <td>347 (3.0)</td>
      <td>92 (3.2)</td>
      <td>255 (2.9)</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>202 (1.7)</td>
      <td>48 (1.7)</td>
      <td>154 (1.8)</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>276 (2.4)</td>
      <td>70 (2.4)</td>
      <td>206 (2.4)</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>10706 (92.5)</td>
      <td>2668 (92.2)</td>
      <td>8038 (92.6)</td>
    </tr>
    <tr>
      <th>Non testable</th>
      <td></td>
      <td>40 (0.3)</td>
      <td>15 (0.5)</td>
      <td>25 (0.3)</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Motor component of the Glasgow Coma Scale, n (%)</th>
      <th>1</th>
      <td>2</td>
      <td>122 (1.1)</td>
      <td>33 (1.1)</td>
      <td>89 (1.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>75 (0.6)</td>
      <td>16 (0.6)</td>
      <td>59 (0.7)</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>74 (0.6)</td>
      <td>26 (0.9)</td>
      <td>48 (0.6)</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>122 (1.1)</td>
      <td>28 (1.0)</td>
      <td>94 (1.1)</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>469 (4.1)</td>
      <td>109 (3.8)</td>
      <td>360 (4.1)</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>10707 (92.5)</td>
      <td>2680 (92.6)</td>
      <td>8027 (92.5)</td>
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
      <td>2</td>
      <td>367 (3.2)</td>
      <td>92 (3.2)</td>
      <td>275 (3.2)</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>224 (1.9)</td>
      <td>50 (1.7)</td>
      <td>174 (2.0)</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>133 (1.1)</td>
      <td>29 (1.0)</td>
      <td>104 (1.2)</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>472 (4.1)</td>
      <td>112 (3.9)</td>
      <td>360 (4.1)</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>10365 (89.6)</td>
      <td>2609 (90.2)</td>
      <td>7756 (89.4)</td>
    </tr>
    <tr>
      <th>Non testable</th>
      <td></td>
      <td>12 (0.1)</td>
      <td>2 (0.1)</td>
      <td>10 (0.1)</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Alert, voice, pain, unresponsive scale, n (%)</th>
      <th>Alert</th>
      <td>3</td>
      <td>10637 (91.9)</td>
      <td>2664 (92.1)</td>
      <td>7973 (91.9)</td>
    </tr>
    <tr>
      <th>Pain responsive</th>
      <td></td>
      <td>610 (5.3)</td>
      <td>133 (4.6)</td>
      <td>477 (5.5)</td>
    </tr>
    <tr>
      <th>Unresponsive</th>
      <td></td>
      <td>112 (1.0)</td>
      <td>35 (1.2)</td>
      <td>77 (0.9)</td>
    </tr>
    <tr>
      <th>Voice responsive</th>
      <td></td>
      <td>213 (1.8)</td>
      <td>61 (2.1)</td>
      <td>152 (1.8)</td>
    </tr>
    <tr>
      <th>Heart rate, median [Q1,Q3]</th>
      <th></th>
      <td>11</td>
      <td>84.0 [76.0,93.0]</td>
      <td>83.0 [76.0,93.0]</td>
      <td>84.0 [77.0,94.0]</td>
    </tr>
    <tr>
      <th>Systolic blood pressure in mmHg, median [Q1,Q3]</th>
      <th></th>
      <td>36</td>
      <td>123.0 [113.0,132.0]</td>
      <td>124.0 [114.0,132.0]</td>
      <td>123.0 [113.0,132.0]</td>
    </tr>
    <tr>
      <th>Diastolic blood pressure in mmHg, median [Q1,Q3]</th>
      <th></th>
      <td>37</td>
      <td>80.0 [72.0,88.0]</td>
      <td>80.0 [72.0,88.0]</td>
      <td>80.0 [72.0,88.0]</td>
    </tr>
    <tr>
      <th>Peripheral capillary oxygen saturation, median [Q1,Q3]</th>
      <th></th>
      <td>9</td>
      <td>98.0 [98.0,98.0]</td>
      <td>98.0 [98.0,98.0]</td>
      <td>98.0 [98.0,98.0]</td>
    </tr>
    <tr>
      <th>Respiratory rate in breaths per minute, median [Q1,Q3]</th>
      <th></th>
      <td>14</td>
      <td>21.0 [20.0,24.0]</td>
      <td>22.0 [20.0,24.0]</td>
      <td>21.0 [20.0,24.0]</td>
    </tr>
    <tr>
      <th>Time between injury and arrival to participating centre in minutes, median [Q1,Q3]</th>
      <th></th>
      <td>87</td>
      <td>0.0 [0.0,0.0]</td>
      <td>0.0 [0.0,0.0]</td>
      <td>0.0 [0.0,0.0]</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Composite outcome, n (%)</th>
      <th>No</th>
      <td>0</td>
      <td>11334 (97.9)</td>
      <td>2834 (97.9)</td>
      <td>8500 (97.9)</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td></td>
      <td>241 (2.1)</td>
      <td>60 (2.1)</td>
      <td>181 (2.1)</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Level of priority assigned by clinicians, n (%)</th>
      <th>Green</th>
      <td>0</td>
      <td>6464 (55.8)</td>
      <td>1605 (55.5)</td>
      <td>4859 (56.0)</td>
    </tr>
    <tr>
      <th>Orange</th>
      <td></td>
      <td>886 (7.7)</td>
      <td>215 (7.4)</td>
      <td>671 (7.7)</td>
    </tr>
    <tr>
      <th>Red</th>
      <td></td>
      <td>538 (4.6)</td>
      <td>137 (4.7)</td>
      <td>401 (4.6)</td>
    </tr>
    <tr>
      <th>Yellow</th>
      <td></td>
      <td>3687 (31.9)</td>
      <td>937 (32.4)</td>
      <td>2750 (31.7)</td>
    </tr>
  </tbody>
</table>
</div><br />




```python
!jupyter nbconvert --to markdown ./reports/manuscript.ipynb
```

    [NbConvertApp] Converting notebook ./reports/manuscript.ipynb to markdown
    [NbConvertApp] Writing 32502 bytes to reports/manuscript.md



```python
!pandoc -s ./reports/manuscript.md -t html -o ./reports/manuscript.html --citeproc --bibliography="./reports/bibliography.bib" --csl="./reports/apa.csl"

```

    [WARNING] This document format requires a nonempty <title> element.
      Defaulting to 'manuscript' as the title.
      To specify a title, use 'title' in metadata or --metadata title="...".



```python
!pandoc -s ./reports/manuscript.html -t docx -o ./reports/manuscript.docx --citeproc --bibliography="./reports/bibliography.bib" --csl="./reports/apa.csl"

```

    [WARNING] Invalid 'lang' value ''.
      Use an IETF language tag like 'en-US'.

