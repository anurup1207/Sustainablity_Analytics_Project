import pulp
import numpy as np
import pandas as pd
from warnings import filterwarnings

filterwarnings("ignore")

vehicles_df = pd.read_csv("../data/Master_data/vehicle_master_data_4.csv")
distances_df = pd.read_csv("../data/Master_data/distance_master_data_2.csv")

distances_df["Source"] = distances_df["Source"].str.strip()
distances_df["Destination"] = distances_df["Destination"].str.strip()
distances_df["Source"] = distances_df["Source"].str.replace(" ", "_")
distances_df["Destination"] = distances_df["Destination"].str.replace(" ", "_")

def clean_data_mode(df):
    # Drop mode distance that are not compatible to the specific vehicle
    df.drop(
        df[(df["Air Distance (km)"].isna()) & (df["Transportation modes"] == "Air")].index,
        axis=0,
        inplace=True,
    )
    df.drop(
        df[(df["Rail Distance (km)"].isna()) & (df["Transportation modes"] == "Rail")].index,
        axis=0,
        inplace=True,
    )
    df.drop(
        df[(df["Road Distance (km)"].isna()) & (df["Transportation modes"] == "Road")].index,
        axis=0,
        inplace=True,
    )
    df.reset_index(inplace=True, drop=True)

    return df


def pre_calculate_data_features(df):
    # Pre-calculate time, cost and obj function value
    time_list = []
    cost_list = []
    obj1_list = []

    for i in range(len(df)):
        if df["Transportation modes"][i] == "Air":
            time_list.append(df["Air Distance (km)"][i] / df["Avg speed (km/hr)"][i])
            cost_list.append(df["Air Distance (km)"][i] * df["Cost per km"][i])
            obj1_list.append(df["Air Distance (km)"][i] * df["Co2e (g/km)"][i] / 1000)

        elif df["Transportation modes"][i] == "Road":
            time_list.append(df["Road Distance (km)"][i] / df["Avg speed (km/hr)"][i])
            cost_list.append(df["Road Distance (km)"][i] * df["Cost per km"][i])
            obj1_list.append(df["Road Distance (km)"][i] * df["Co2e (g/km)"][i] / 1000)

        else:
            time_list.append(df["Rail Distance (km)"][i] / df["Avg speed (km/hr)"][i])
            cost_list.append(df["Rail Distance (km)"][i] * df["Cost per km"][i])
            obj1_list.append(df["Rail Distance (km)"][i] * df["Co2e (g/km)"][i] / 1000)

    df["Time (hrs)"] = time_list
    df["Cost"] = cost_list
    df["obj1"] = obj1_list


def prepare_model_data(vehicles_df, distances_df):
    # Creating decision variable column in the dataframe
    df = distances_df.join(vehicles_df, how="cross")
    df["decisions"] = df["Source"] + ", " + df["Destination"] + ", " + df["Transportation modes"] + ", " + df["Vehicle_type"]

    df = clean_data_mode(df)

    pre_calculate_data_features(df)

    return df


def master_data_preparation(vehicles_df, demands_df):
    # master availability mapping
    master_availability = dict(zip(vehicles_df["Vehicle_type"], vehicles_df["Availability"]))

    # master demand mapping
    master_demand = {}
    for i, j in zip(
        np.array(demands_df[["Source", "Destination"]].drop_duplicates()),
        [
            0,
        ]
        * len(demands_df[["Source", "Destination"]].drop_duplicates()),
    ):
        master_demand[tuple(i)] = j

    return master_availability, master_demand


def lp_expression_to_dict(expression):
    return {
        'name': expression.name,
        'coefficients': {var.name: coeff for var, coeff in expression.items()},
        'constant': expression.constant,
    }


df0 = prepare_model_data(vehicles_df, distances_df)
master_availability, master_demand = master_data_preparation(vehicles_df, df0)

choice_decision = pulp.LpVariable.dicts(
        "Choice_Decision_",
        (
            tuple(df0["decisions"][i].split(", ")) + (j,)
            for i in range(len(df0))
            for j in range(master_availability[df0["decisions"][i].split(", ")[-1]])
        ),
        lowBound=0,
        cat="Binary",
    )

# trip/return trip decision variable -> Int
trip_decision = pulp.LpVariable.dicts(
    "Trip_Decision_",
    (
        tuple(df0["decisions"][i].split(", ")) + (j,)
        for i in range(len(df0))
        for j in range(master_availability[df0["decisions"][i].split(", ")[-1]])
    ),
    lowBound=0,
    cat="Integer",
)

decision_df = pd.DataFrame(list(choice_decision.keys()), columns=['Source', 'Destination', 'Transportation modes', 'Vehicle_type', 'Vehicle number'])
decision_df['Choice decision'] = pd.DataFrame(list(choice_decision.values()))
decision_df['Trip decision'] = pd.DataFrame(list(trip_decision.values()))
decision_df['S_D'] = tuple(zip(decision_df['Source'], decision_df['Destination']))

ddf = decision_df.merge(df0, on=['Source', 'Destination', 'Transportation modes', 'Vehicle_type'], how='left')

demand_decision_df = ddf[['Source', 'Destination', 'S_D', 'Capacity (metric tons)', 'Choice decision', 'Trip decision']]
moq_decision_df = ddf[['Source', 'Destination', 'S_D', 'Minimum Quantity Allowed (MQA)', 'Choice decision', 'Trip decision']]
time_decision_df = ddf[['Source', 'Destination', 'S_D', 'Time (hrs)', 'Choice decision', 'Trip decision']]
cost_decision_df = ddf[['Source', 'Destination', 'S_D', 'Cost', 'Choice decision', 'Trip decision']]
emission_decision_df = ddf[['Source', 'Destination', 'S_D', 'Co2e (g/km)', 'Choice decision', 'Trip decision']]
obj_df = ddf[['Source', 'Destination', 'S_D', 'obj1', 'Choice decision', 'Trip decision']]

demand_decision_df['Choice decision'] = demand_decision_df['Choice decision']*demand_decision_df['Capacity (metric tons)']
demand_decision_df['Trip decision'] = demand_decision_df['Trip decision']*demand_decision_df['Capacity (metric tons)']
demand_decision_df = demand_decision_df.groupby(['Source', 'Destination', 'S_D']).sum()[['Choice decision', 'Trip decision']].reset_index()

moq_decision_df['Choice decision'] = moq_decision_df['Choice decision']*moq_decision_df['Minimum Quantity Allowed (MQA)']
moq_decision_df['Trip decision'] = moq_decision_df['Trip decision']*moq_decision_df['Minimum Quantity Allowed (MQA)']
moq_decision_df = moq_decision_df.groupby(['Source', 'Destination', 'S_D']).sum()[['Choice decision', 'Trip decision']].reset_index()

time_decision_df['Choice decision'] = time_decision_df['Choice decision']*time_decision_df['Time (hrs)']
time_decision_df['Trip decision'] = time_decision_df['Trip decision']*time_decision_df['Time (hrs)']

cost_decision_df['Choice decision'] = cost_decision_df['Choice decision']*cost_decision_df['Cost']
cost_decision_df['Trip decision'] = cost_decision_df['Trip decision']*cost_decision_df['Cost']
cost_decision_df2 = cost_decision_df.groupby(['Source', 'Destination', 'S_D']).sum()[['Choice decision', 'Trip decision']].reset_index()

emission_decision_df['Choice decision'] = emission_decision_df['Choice decision']*emission_decision_df['Co2e (g/km)']
emission_decision_df['Trip decision'] = emission_decision_df['Trip decision']*emission_decision_df['Co2e (g/km)']

obj_df['Choice decision'] = obj_df['Choice decision']*obj_df['obj1']
obj_df['Trip decision'] = obj_df['Trip decision']*obj_df['obj1']

if __name__ == "__main__":
    print("Done!")
