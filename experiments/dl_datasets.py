from folktables import ACSDataSource, ACSIncome, generate_categories
import DP as dp
from corels import load_from_csv, RuleList, CorelsClassifier

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=["CA"], download=True)


definition_df = data_source.get_definitions(download=True)
categories = generate_categories(features=ACSIncome.features, definition_df=definition_df)

dataset, ca_labels, _ = ACSIncome.df_to_pandas(ca_data, categories=categories, dummies=True)


print(dataset.head)
print(ca_labels.head)


rmv_folktable = ["RAC1P", "SEX", "POBP", "OCCP", "AGEP"]

for feature in rmv_folktable : 
    dataset = dataset.loc[:,~dataset.columns.str.startswith(feature)]
    
    
dataset = dataset.dropna()

dataset['WKHP_high'] = dataset['WKHP'].apply(lambda x: x >= 50)
dataset['WKHP_low'] = dataset['WKHP'].apply(lambda x: x <=30)

dataset = dataset.drop("WKHP", axis = 1)


for column in dataset:
    new_name = column
    new_name= column.replace('''"''',"")
    new_name=new_name.replace(",","")
    dataset=dataset.rename(columns={column: new_name})

dataset = dataset.join(ca_labels)
dataset= dataset.rename(columns={"PINCP" : "Wage>50k"})
dataset = dataset.astype(int)


print(dataset.head)

dataset.to_csv('data/folktable.csv', index=False)

