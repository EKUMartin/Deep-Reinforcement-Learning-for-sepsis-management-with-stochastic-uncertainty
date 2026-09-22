import pandas as pd
import numpy as np
from tqdm import tqdm


class target_cohort:
    def __init__(self, withclause, fromwhereclause, conn, cur):
        self.withclause = withclause
        self.fromwhereclause = fromwhereclause
        self.conn = conn
        self.cur = cur

    def query(self):
        q = f"""
        {self.withclause}
        SELECT subject_id, stay_id, intime
        {self.fromwhereclause};
        """

        self.cur.execute(q)
        q_stay_id_result = self.cur.fetchall()

        first_stay_id = []

        for i in q_stay_id_result:
            first_stay_id.append(i['stay_id'])

        return first_stay_id


class states_preprocessor:
    def __init__(
        self,
        conn,
        cur,
        INTERVAL,
        my_required_items,
        first_stay_id,
        initial_values,
        zero_fill_cols
    ):
        self.conn = conn
        self.cur = cur
        self.INTERVAL = INTERVAL
        self.initial_values = initial_values
        self.first_stay_id = first_stay_id
        self.my_required_items = my_required_items
        self.zero_fill_cols = zero_fill_cols

    def query(self, config, stay_id_list, conn):
        stay_str = ','.join(map(str, stay_id_list))

        sql = f"""
            WITH item_filtered AS (
                SELECT
                    {config['stay_id_col']} AS stay_id,
                    {config['time_col']} AS charttime,
                    {config['value_col']} AS {config['item_name']}
                FROM {config['table_name']}
                {config['join_clause']}
                WHERE itemid = {config['item_id']}
            )
            SELECT
                stay_id,
                charttime,
                {config['item_name']}
            FROM item_filtered
            WHERE stay_id IN ({stay_str})
        """

        return pd.read_sql(sql, conn)

    def remove_outliers(self, config, data):
        item = config['item_name']

        if not data.empty and item in data.columns:
            q_low = data.groupby('stay_id')[item].transform(
                lambda x: x.quantile(0.01)
            )

            q_hi = data.groupby('stay_id')[item].transform(
                lambda x: x.quantile(0.99)
            )

            data = data[
                (data[item] >= q_low) &
                (data[item] <= q_hi)
            ]

        return data

    def resampling(self, config, data):
        if data.empty:
            return data

        data['charttime'] = pd.to_datetime(
            data['charttime']
        )

        resampled = (
            data.groupby('stay_id')
            .apply(
                lambda x:
                x.set_index('charttime')
                .resample(
                    f"{config['resampling_hour']}h"
                )[config['item_name']]
                .agg(config['resample_method'])
            )
            .reset_index()
        )

        return resampled

    def preprocessing(
        self,
        config,
        data,
        initial_values
    ):
        item = config['item_name']
        method = config['fill_method']

        if data.empty or item not in data.columns:
            return data

        data[item] = pd.to_numeric(
            data[item],
            errors='coerce'
        )

        if initial_values and item in initial_values:
            first_indices = (
                data.groupby('stay_id')
                .head(1)
                .index
            )

            data.loc[first_indices, item] = (
                data.loc[first_indices, item]
                .fillna(initial_values[item])
            )

        if method == 'ffill':
            data[item] = (
                data.groupby('stay_id')[item]
                .ffill()
                .bfill()
            )

        elif method == 'bfill':
            data[item] = (
                data.groupby('stay_id')[item]
                .bfill()
                .ffill()
            )

        elif method == 'interpolate':
            data[item] = (
                data.groupby('stay_id')[item]
                .transform(
                    lambda x:
                    x.interpolate()
                    .ffill()
                    .bfill()
                )
            )

        return data

    def create_pipeline_config(
        self,
        required_items,
        global_resample_hour=1
    ):
        config_list = []

        for item in required_items:
            config = {
                'item_id': item['item_id'],
                'item_name': item['item_name'],
                'table_name': item.get(
                    'table_name',
                    'mimic.chartevents'
                ),
                'time_col': item.get(
                    'time_col',
                    'charttime'
                ),
                'value_col': item.get(
                    'value_col',
                    'valuenum'
                ),
                'stay_id_col': item.get(
                    'stay_id_col',
                    'stay_id'
                ),
                'join_clause': item.get(
                    'join_clause',
                    ''
                ),
                'resampling_hour': item.get(
                    'resampling_hour',
                    global_resample_hour
                ),
                'resample_method': item.get(
                    'resample_method',
                    'mean'
                ),
                'fill_method': item.get(
                    'fill_method',
                    'interpolate'
                )
            }

            config_list.append(config)

        return config_list

    def get_data(
        self,
        config_list,
        stay_id_list,
        conn,
        initial_values
    ):
        extracted_data = {}

        for config in tqdm(
            config_list,
            desc="Extracting & Preprocessing Features"
        ):
            raw_df = self.query(
                config,
                stay_id_list,
                conn
            )

            if raw_df.empty:
                extracted_data[
                    config['item_name']
                ] = raw_df
                continue

            filtered_df = self.remove_outliers(
                config,
                raw_df
            )

            resampled_df = self.resampling(
                config,
                filtered_df
            )

            preprocessed_df = self.preprocessing(
                config,
                resampled_df,
                initial_values
            )

            extracted_data[
                config['item_name']
            ] = preprocessed_df

        final_df = None

        for _, df in extracted_data.items():
            if df.empty:
                continue

            if final_df is None:
                final_df = df

            else:
                final_df = pd.merge(
                    final_df,
                    df,
                    on=['stay_id', 'charttime'],
                    how='outer'
                )

        if final_df is None:
            return pd.DataFrame()

        final_df = (
            final_df
            .sort_values(
                ['stay_id', 'charttime']
            )
            .reset_index(drop=True)
        )

        if (
            'BPM_inv' in final_df.columns
            and 'BPM_noninv' in final_df.columns
        ):
            final_df['heart_rate'] = (
                final_df['BPM_inv']
                .fillna(
                    final_df['BPM_noninv']
                )
            )

            final_df = final_df.drop(
                columns=[
                    'BPM_inv',
                    'BPM_noninv'
                ]
            )

        elif 'BPM_inv' in final_df.columns:
            final_df['heart_rate'] = (
                final_df['BPM_inv']
            )

            final_df = final_df.drop(
                columns=['BPM_inv']
            )

        elif 'BPM_noninv' in final_df.columns:
            final_df['heart_rate'] = (
                final_df['BPM_noninv']
            )

            final_df = final_df.drop(
                columns=['BPM_noninv']
            )

        final_df['SIRS'] = 0
        final_df['shock_index'] = 0.0

        sirs_cols = [
            'Temperature',
            'heart_rate',
            'RR',
            'WBC'
        ]

        if all(
            col in final_df.columns
            for col in sirs_cols
        ):
            final_df['SIRS'] = final_df.apply(
                self.calculate_sirs,
                axis=1
            )

        shock_cols = [
            'heart_rate',
            'NIBPs'
        ]

        if all(
            col in final_df.columns
            for col in shock_cols
        ):
            final_df['shock_index'] = (
                final_df.apply(
                    self.shock_index,
                    axis=1
                )
            )

        return final_df

    def calculate_sirs(self, row):
        count = 0

        if (
            pd.notna(row.get('Temperature'))
            and (
                row['Temperature'] > 38
                or row['Temperature'] < 36
            )
        ):
            count += 1

        if (
            pd.notna(row.get('heart_rate'))
            and row['heart_rate'] > 90
        ):
            count += 1

        if (
            pd.notna(row.get('RR'))
            and row['RR'] > 20
        ):
            count += 1

        if (
            pd.notna(row.get('WBC'))
            and (
                row['WBC'] > 12000
                or row['WBC'] < 4000
            )
        ):
            count += 1

        return count if count >= 2 else 0

    def shock_index(self, row):
        hr = row.get('heart_rate')
        sbp = row.get('NIBPs')

        if (
            pd.isna(hr)
            or pd.isna(sbp)
            or sbp == 0
        ):
            return np.nan

        return hr / sbp

    def fill_zero(
        self,
        df,
        zero_fill_cols
    ):
        for col in zero_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def get_ages(
        self,
        first_stay_id_str,
        cur
    ):
        sql_age = f"""
            WITH group_age AS (
                SELECT
                    i.stay_id,
                    (
                        p.anchor_age +
                        (
                            EXTRACT(
                                YEAR FROM a.admittime
                            ) - p.anchor_year
                        )
                    ) AS age
                FROM mimic.icustays i
                LEFT JOIN mimic_hosp.admissions a
                    ON i.hadm_id = a.hadm_id
                LEFT JOIN mimic.patients p
                    ON i.subject_id = p.subject_id
            )
            SELECT
                group_age.stay_id,
                group_age.age
            FROM group_age
            WHERE group_age.stay_id IN (
                {first_stay_id_str}
            )
            ORDER BY group_age.age DESC;
        """

        cur.execute(sql_age)
        age_result = cur.fetchall()

        return pd.DataFrame(age_result)

    def main(self):
        config_list = (
            self.create_pipeline_config(
                self.my_required_items,
                global_resample_hour=self.INTERVAL
            )
        )

        final_dataframe = self.get_data(
            config_list,
            self.first_stay_id,
            self.conn,
            self.initial_values
        )

        final_dataframe = self.fill_zero(
            final_dataframe,
            self.zero_fill_cols
        )

        return final_dataframe


class action_preprocessor:
    def __init__(
        self,
        conn,
        cur,
        resample_hour,
        stay_id_list
    ):
        self.conn = conn
        self.cur = cur
        self.stay_id_list = stay_id_list
        self.resample_hour = resample_hour

    def get_action_data(self, state_grid):
        stay_str = ','.join(
            map(str, self.stay_id_list)
        )

        query = f"""
            SELECT *
            FROM value_based_data.treatment_features
            WHERE stay_id IN ({stay_str})
            ORDER BY stay_id, time_hour
        """

        action_df = pd.read_sql(
            query,
            con=self.conn
        )

        if action_df.empty:
            return pd.DataFrame()

        action_df['time_hour'] = pd.to_datetime(
            action_df['time_hour'],
            errors='coerce'
        )

        action_df = action_df.dropna(
            subset=['time_hour']
        ).copy()

        if 'charttime' in action_df.columns:
            action_df = action_df.drop(
                columns=['charttime']
            )

        state_grid = state_grid[
            ['stay_id', 'charttime']
        ].copy()

        state_grid['charttime'] = pd.to_datetime(
            state_grid['charttime']
        )

        state_grid = (
            state_grid
            .drop_duplicates(
                subset=[
                    'stay_id',
                    'charttime'
                ]
            )
        )

        action_df = (
            action_df
            .sort_values(
                ['time_hour', 'stay_id']
            )
            .reset_index(drop=True)
        )

        state_grid = (
            state_grid
            .sort_values(
                ['charttime', 'stay_id']
            )
            .reset_index(drop=True)
        )

        mapped = pd.merge_asof(
            action_df,
            state_grid,
            left_on='time_hour',
            right_on='charttime',
            by='stay_id',
            direction='forward',
            allow_exact_matches=True
        )

        mapped = mapped.dropna(
            subset=['charttime']
        ).copy()

        vasopressors = [
            'norepinephrine',
            'dopamine',
            'epinephrine',
            'phenylephrine',
            'vasopressin',
            'dobutamine'
        ]

        vaso_amounts = [
            v + '_amount'
            for v in vasopressors
        ]

        excluded_base = [
            'angiotensin_ii',
            'Phenylephrine50250',
            'Phenylephrine200250_old',
            'Phenylephrine200250'
        ]

        excluded_amounts = [
            x + '_amount'
            for x in excluded_base
        ]

        excluded_cols = set(
            vasopressors
            + vaso_amounts
            + excluded_base
            + excluded_amounts
            + [
                'stay_id',
                'time_hour',
                'charttime',
                'time_bin',
                'event_id',
                'patient_weight',
                'vaso',
                'iv_fluid',
                'iv_action',
                'vaso_action',
                'final_action'
            ]
        )

        iv_cols = []

        for col in mapped.columns:
            if col in excluded_cols:
                continue

            if pd.api.types.is_numeric_dtype(
                mapped[col]
            ):
                iv_cols.append(col)

        agg_dict = {}

        for vaso_col in vasopressors:
            if vaso_col in mapped.columns:
                agg_dict[vaso_col] = 'max'

        if 'patient_weight' in mapped.columns:
            agg_dict['patient_weight'] = 'mean'

        for col in iv_cols:
            agg_dict[col] = 'sum'

        grouped = (
            mapped
            .groupby(
                ['stay_id', 'charttime'],
                as_index=False
            )
            .agg(agg_dict)
        )

        grouped['vaso'] = grouped.apply(
            self.cal_vasopressor_action,
            axis=1
        )

        if len(iv_cols) > 0:
            grouped['iv_fluid'] = (
                grouped[iv_cols]
                .sum(axis=1)
            )
        else:
            grouped['iv_fluid'] = 0.0

        grouped['iv_action'] = (
            grouped['iv_fluid']
            .apply(
                self.discretize_iv_fluid
            )
        )

        grouped['vaso_action'] = (
            grouped['vaso']
            .apply(
                self.discretize_vasopressor
            )
        )

        grouped['final_action'] = (
            (grouped['iv_action'] - 1) * 5
            + (grouped['vaso_action'] - 1)
        )

        return grouped[
            [
                'stay_id',
                'charttime',
                'iv_fluid',
                'vaso',
                'iv_action',
                'vaso_action',
                'final_action'
            ]
        ].copy()

    def discretize_iv_fluid(self, val):
        if val == 0:
            return 1

        if 0 < val <= 50:
            return 2

        if 50 < val <= 180:
            return 3

        if 180 < val <= 530:
            return 4

        return 5

    def discretize_vasopressor(self, val):
        if val == 0:
            return 1

        if 0 < val <= 0.08:
            return 2

        if 0.08 < val <= 0.22:
            return 3

        if 0.22 < val <= 0.45:
            return 4

        return 5

    def cal_vasopressor_action(self, row):
        norepi = row.get(
            'norepinephrine',
            0
        )

        dopa = row.get(
            'dopamine',
            0
        )

        epi = row.get(
            'epinephrine',
            0
        )

        phenyl = row.get(
            'phenylephrine',
            0
        )

        vaso = row.get(
            'vasopressin',
            0
        )

        return (
            norepi
            + (1 / 150) * dopa
            + 0.1 * epi
            + 0.1 * phenyl
            + (2.5 * vaso) / 60
        )

    def main(self, state_grid):
        return self.get_action_data(
            state_grid
        )

class sofa:
    def __init__(
        self,
        conn,
        cur,
        stay_ids
    ):
        self.conn = conn
        self.cur = cur
        self.stay_ids = stay_ids

    def main(self):
        stay_str = ','.join(
            map(str, self.stay_ids)
        )

        q = f"""
            SELECT
                stay_id,
                chart_hour,
                total_sofa_score
            FROM hrl.sofa_score_test
            WHERE stay_id IN ({stay_str})
        """

        return pd.read_sql(
            q,
            self.conn
        )


class survival_labeler:
    def __init__(
        self,
        conn,
        cur,
        stay_ids
    ):
        self.conn = conn
        self.cur = cur
        self.stay_ids = stay_ids

    def check_survival(
        self,
        row
    ):
        if not row['is_last_stay']:
            return 0

        if (
            pd.notna(row['dischtime'])
            and pd.notna(row['deathtime'])
            and row['dischtime']
            == row['deathtime']
        ):
            return 1

        if (
            pd.notna(row['dod'])
            and pd.notna(row['dischtime'])
        ):
            days_to_death = (
                row['dod']
                - row['dischtime']
            ).days

            if 0 <= days_to_death <= 90:
                return 2

        return 0

    def main(self):
        stay_ids_clean = [
            str(int(x))
            for x in self.stay_ids
        ]

        stay_ids_joined = ', '.join(
            stay_ids_clean
        )

        stay_ids_sql = (
            f'({stay_ids_joined})'
        )

        patient_query = f"""
            SELECT
                i.stay_id,
                i.subject_id,
                p.gender,
                (
                    p.anchor_age +
                    (
                        EXTRACT(
                            YEAR FROM a.admittime
                        ) - p.anchor_year
                    )
                ) AS age,
                a.race,
                a.dischtime,
                a.deathtime,
                p.dod
            FROM mimic.icustays i
            LEFT JOIN mimic_hosp.admissions a
                ON i.hadm_id = a.hadm_id
            LEFT JOIN mimic.patients p
                ON i.subject_id = p.subject_id
            WHERE i.stay_id IN {stay_ids_sql};
        """

        self.cur.execute(
            patient_query
        )

        cols = [
            desc[0]
            for desc in self.cur.description
        ]

        patients_df = pd.DataFrame(
            self.cur.fetchall(),
            columns=cols
        )

        patients_df['dischtime'] = (
            pd.to_datetime(
                patients_df['dischtime']
            )
        )

        patients_df['deathtime'] = (
            pd.to_datetime(
                patients_df['deathtime']
            )
        )

        patients_df['dod'] = (
            pd.to_datetime(
                patients_df['dod']
            )
        )

        patients_df = (
            patients_df
            .sort_values(
                by=[
                    'subject_id',
                    'dischtime'
                ]
            )
        )

        patients_df[
            'is_last_stay'
        ] = ~patients_df.duplicated(
            subset=['subject_id'],
            keep='last'
        )

        patients_df[
            'survival'
        ] = patients_df.apply(
            self.check_survival,
            axis=1
        )

        return patients_df[
            ['stay_id', 'survival']
        ]