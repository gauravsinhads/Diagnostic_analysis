import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="TPA Recruitment Analysis Dashboard")

# --- Helper Functions ---

def calculate_rate(df, score_threshold, comparison, status_to_count):
    """Calculates the pass or fail rate for a given score threshold."""
    if df.empty: return "N/A"
    if comparison == 'above':
        subset = df[df['TALKSCORE_OVERALL'] >= score_threshold]
    else:  # below
        subset = df[df['TALKSCORE_OVERALL'] < score_threshold]

    denominator = subset[subset['Pass/Fail Status'].isin(['Passed HM', 'Failed HM'])]
    numerator = denominator[denominator['Pass/Fail Status'] == status_to_count]

    if len(denominator) == 0:
        return "N/A"

    rate = (len(numerator) / len(denominator)) * 100
    return f"{rate:.1f}%"

def calculate_counts(df, score_threshold, comparison):
    """Calculates the pass and fail counts for a given score threshold."""
    if df.empty: return 0, 0
    if comparison == 'above':
        subset = df[df['TALKSCORE_OVERALL'] >= score_threshold]
    else:  # below
        subset = df[df['TALKSCORE_OVERALL'] < score_threshold]

    denominator_df = subset[subset['Pass/Fail Status'].isin(['Passed HM', 'Failed HM'])]
    pass_count = len(denominator_df[denominator_df['Pass/Fail Status'] == 'Passed HM'])
    fail_count = len(denominator_df[denominator_df['Pass/Fail Status'] == 'Failed HM'])

    return pass_count, fail_count

# --- Main App Logic ---

st.title("Recruitment Funnel Analysis Dashboard")

try:
    # Load the data directly from the CSV file
    tpa = pd.read_csv('tpa.csv')

    # --- Global Data Preparation ---
    tpa['INVITATIONDT'] = pd.to_datetime(tpa['INVITATIONDT'], errors='coerce')
    tpa.dropna(subset=['INVITATIONDT'], inplace=True)
    
    # --- Filters Section ---
    st.header("Filters")
    
    # Filter 1: Date range for 'INVITATIONDT'
    min_date = tpa['INVITATIONDT'].min().date()
    max_date = tpa['INVITATIONDT'].max().date()
    
    default_start_date = max_date - timedelta(days=60)
    if default_start_date < min_date:
        default_start_date = min_date

    start_date, end_date = st.date_input(
        "Invitation Date Range",
        [default_start_date, max_date],
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed"
    )
    st.divider()

    # Filter 2: Expander for 'CAMPAIGN_SITE' with Select All
    with st.expander("Select Campaign Site(s)", expanded=True):
        unique_sites = sorted(tpa['CAMPAIGN_SITE'].dropna().unique())
        select_all_sites = st.checkbox("Select All Sites", value=True, key='sites_select_all')
        default_selection_sites = unique_sites if select_all_sites else []
        selected_sites = st.multiselect(
            "Campaign Site", options=unique_sites,
            default=default_selection_sites,
            label_visibility="collapsed"
        )
    st.divider()

    # Filter 3: Dependent Expander for 'CAMPAIGNTITLE' with Select All
    with st.expander("Select Campaign Title(s)", expanded=True):
        if not selected_sites:
            st.warning("Please select a Campaign Site to see available titles.")
            selected_titles = []
        else:
            available_titles = sorted(tpa[tpa['CAMPAIGN_SITE'].isin(selected_sites)]['CAMPAIGNTITLE'].dropna().unique())
            select_all_titles = st.checkbox("Select All Titles", value=True, key='titles_select_all')
            default_selection_titles = available_titles if select_all_titles else []
            selected_titles = st.multiselect(
                "Campaign Title", options=available_titles,
                default=default_selection_titles,
                label_visibility="collapsed"
            )
    st.divider()

    # --- Data Filtering Logic ---
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    filtered_tpa = tpa[
        (tpa['INVITATIONDT'] >= start_datetime) &
        (tpa['INVITATIONDT'] <= end_datetime) &
        (tpa['CAMPAIGN_SITE'].isin(selected_sites)) &
        (tpa['CAMPAIGNTITLE'].isin(selected_titles))
    ].copy()
    
    if filtered_tpa.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
    else:
        # --- Data Preparation on Filtered Data ---
        filtered_tpa['Month_Year'] = filtered_tpa['INVITATIONDT'].dt.strftime('%Y-%m')
        conditions = [
            filtered_tpa['LABELS'].str.contains('Passed HM', na=False),
            (filtered_tpa['LABELS'].str.contains('Failed HM', na=False)) & (~filtered_tpa['LABELS'].str.contains('Passed HM', na=False))
        ]
        choices = ['Passed HM', 'Failed HM']
        filtered_tpa['Pass/Fail Status'] = np.select(conditions, choices, default='Not Applicable')
        filtered_tpa['TALKSCORE_OVERALL'] = pd.to_numeric(filtered_tpa['TALKSCORE_OVERALL'], errors='coerce')


        # --- Output 1: Monthly Pass vs. Fail for B2-C2 Candidates ---
        st.header("Monthly Pass vs. Fail Analysis for B2-C2 Candidates")
        
        cefr_levels_to_include = ['B2', 'B2+', 'C1', 'C2']
        tpa_filtered_cefr = filtered_tpa[filtered_tpa['CEFR'].isin(cefr_levels_to_include)].copy()

        if tpa_filtered_cefr.empty:
            st.info("No B2-C2 level candidates found for the selected filters.")
        else:
            # Calculations for graphs
            monthly_total = tpa_filtered_cefr.groupby('Month_Year')['CAMPAIGNINVITATIONID'].nunique()
            passed_df = tpa_filtered_cefr[tpa_filtered_cefr['Pass/Fail Status'] == 'Passed HM']
            monthly_passed = passed_df.groupby('Month_Year')['CAMPAIGNINVITATIONID'].nunique()
            monthly_pass_percentage = (monthly_passed / monthly_total * 100).fillna(0)
            failed_df = tpa_filtered_cefr[tpa_filtered_cefr['Pass/Fail Status'] == 'Failed HM']
            monthly_failed = failed_df.groupby('Month_Year')['CAMPAIGNINVITATIONID'].nunique()
            monthly_fail_percentage = (monthly_failed / monthly_total * 100).fillna(0)

            plot_data_pass = monthly_pass_percentage.reset_index()
            plot_data_pass.columns = ['Month_Year', 'Pass_Percentage']
            plot_data_pass = plot_data_pass.sort_values('Month_Year')
            plot_data_fail = monthly_fail_percentage.reset_index()
            plot_data_fail.columns = ['Month_Year', 'Fail_Percentage']
            plot_data_fail = plot_data_fail.sort_values('Month_Year')

            total_candidates = tpa_filtered_cefr['CAMPAIGNINVITATIONID'].nunique()
            total_passed = passed_df['CAMPAIGNINVITATIONID'].nunique()
            overall_pass_percentage = (total_passed / total_candidates * 100) if total_candidates > 0 else 0
            total_failed = failed_df['CAMPAIGNINVITATIONID'].nunique()
            overall_fail_percentage = (total_failed / total_candidates * 100) if total_candidates > 0 else 0

            # Percentage line graph
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=plot_data_pass['Month_Year'], y=plot_data_pass['Pass_Percentage'], mode='lines+markers+text', name='Monthly Passed HM %', line=dict(color='royalblue', width=2), marker=dict(color='navy', size=8), text=[f'{p:.1f}%' for p in plot_data_pass['Pass_Percentage']], textposition="top center"))
            fig_line.add_trace(go.Scatter(x=plot_data_fail['Month_Year'], y=plot_data_fail['Fail_Percentage'], mode='lines+markers+text', name='Monthly Failed HM %', line=dict(color='indianred', width=2), marker=dict(color='darkred', size=8), text=[f'{p:.1f}%' for p in plot_data_fail['Fail_Percentage']], textposition="bottom center"))
            fig_line.add_hline(y=overall_pass_percentage, line_width=3, line_dash="dash", line_color="royalblue", annotation_text=f"Overall Pass Avg: {overall_pass_percentage:.1f}%", annotation_position="bottom right", annotation_font=dict(size=14, color="royalblue"))
            fig_line.add_hline(y=overall_fail_percentage, line_width=3, line_dash="dash", line_color="firebrick", annotation_text=f"Overall Fail Avg: {overall_fail_percentage:.1f}%", annotation_position="top right", annotation_font=dict(size=14, color="firebrick"))
            fig_line.update_layout(title=dict(text="Monthly Pass vs. Fail Percentage", x=0.5), xaxis_title="Month", yaxis_title="Percentage of Candidates (%)", yaxis=dict(ticksuffix='%'), xaxis=dict(type='category'), plot_bgcolor='white', paper_bgcolor='white', font=dict(family="Arial, sans-serif", size=12, color="black"), showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig_line, use_container_width=True)

            # Absolute numbers bar chart
            counts_df = pd.DataFrame({'Passed': monthly_passed, 'Failed': monthly_failed}).fillna(0).reset_index()
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=counts_df['Month_Year'], y=counts_df['Passed'], name='Passed HM', marker_color='royalblue', text=counts_df['Passed'], textposition='auto'))
            fig_bar.add_trace(go.Bar(x=counts_df['Month_Year'], y=counts_df['Failed'], name='Failed HM', marker_color='indianred', text=counts_df['Failed'], textposition='auto'))
            fig_bar.update_layout(barmode='group', title=dict(text="Monthly Pass vs. Fail Counts", x=0.5), xaxis_title="Month", yaxis_title="Number of Candidates", xaxis=dict(type='category'), plot_bgcolor='white', paper_bgcolor='white', font=dict(family="Arial, sans-serif", size=12, color="black"), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Background Calculation for Subsequent Tables ---
        t1_failed_scores = filtered_tpa[filtered_tpa['LABELS'].str.contains('T1 Failed', na=False)]['TALKSCORE_OVERALL'].dropna()
        t2_failed_scores = filtered_tpa[filtered_tpa['LABELS'].str.contains('T2 Failed', na=False)]['TALKSCORE_OVERALL'].dropna()
        t3_failed_scores = filtered_tpa[filtered_tpa['LABELS'].str.contains('T3 Failed', na=False)]['TALKSCORE_OVERALL'].dropna()
        passed_hm_scores = filtered_tpa[filtered_tpa['Pass/Fail Status'] == 'Passed HM']['TALKSCORE_OVERALL'].dropna()

        stats_passed = passed_hm_scores.describe()
        stats_t1 = t1_failed_scores.describe()
        stats_t2 = t2_failed_scores.describe()
        stats_t3 = t3_failed_scores.describe()

        summary_df_overall = pd.DataFrame({'Passed HM': stats_passed, 'T1 Failed': stats_t1, 'T2 Failed': stats_t2, 'T3 Failed': stats_t3}).T
        summary_df_overall.index.name = 'Category'
        summary_df_overall = summary_df_overall.rename(columns={'count': 'Count', 'mean': 'Mean', 'std': 'Std Dev', 'min': 'Min', '25%': '25% (Q1)', '50%': 'Median', '75%': '75% (Q3)', 'max': 'Max'})
        summary_df_overall['Count'] = summary_df_overall['Count'].astype(int)
        for col in summary_df_overall.columns:
            if col != 'Count':
                summary_df_overall[col] = summary_df_overall[col].round(2)
        summary_df_overall.reset_index(inplace=True)

        # --- Output 2: Overall Summary Statistics ---
        st.header("Overall Summary Statistics for TALKSCORE_OVERALL (Filtered)")
        st.dataframe(summary_df_overall)

        # --- Outputs 3 & 4: Global Analysis Tables ---
        st.header("Global Analysis Based on Score Thresholds (using Filtered Data)")

        categories = ['Passed HM', 'T1 Failed', 'T2 Failed', 'T3 Failed']
        global_median_pct_data = []
        global_q3_pct_data = []
        global_median_abs_data = []
        global_q3_abs_data = []
        stats_df_overall_indexed = summary_df_overall.set_index('Category')

        for category in categories:
            if category in stats_df_overall_indexed.index:
                median_score_threshold = stats_df_overall_indexed.loc[category, 'Median']
                q3_score_threshold = stats_df_overall_indexed.loc[category, '75% (Q3)']

                if pd.notna(median_score_threshold):
                    global_median_pct_data.append({'Category Threshold': category, 'Value': median_score_threshold, 'Global HM Pass Rate % Above': calculate_rate(filtered_tpa, median_score_threshold, 'above', 'Passed HM'), 'Global HM Fail Rate % Above': calculate_rate(filtered_tpa, median_score_threshold, 'above', 'Failed HM'), 'Global HM Pass Rate % Below': calculate_rate(filtered_tpa, median_score_threshold, 'below', 'Passed HM'), 'Global HM Fail Rate % Below': calculate_rate(filtered_tpa, median_score_threshold, 'below', 'Failed HM')})
                    pass_above, fail_above = calculate_counts(filtered_tpa, median_score_threshold, 'above')
                    pass_below, fail_below = calculate_counts(filtered_tpa, median_score_threshold, 'below')
                    global_median_abs_data.append({'Category Threshold': category, 'Value': median_score_threshold, 'Global HM Pass Count Above': pass_above, 'Global HM Fail Count Above': fail_above, 'Global HM Pass Count Below': pass_below, 'Global HM Fail Count Below': fail_below})

                if pd.notna(q3_score_threshold):
                    global_q3_pct_data.append({'Category Threshold': category, 'Value': q3_score_threshold, 'Global HM Pass Rate % Above': calculate_rate(filtered_tpa, q3_score_threshold, 'above', 'Passed HM'), 'Global HM Fail Rate % Above': calculate_rate(filtered_tpa, q3_score_threshold, 'above', 'Failed HM'), 'Global HM Pass Rate % Below': calculate_rate(filtered_tpa, q3_score_threshold, 'below', 'Passed HM'), 'Global HM Fail Rate % Below': calculate_rate(filtered_tpa, q3_score_threshold, 'below', 'Failed HM')})
                    pass_above, fail_above = calculate_counts(filtered_tpa, q3_score_threshold, 'above')
                    pass_below, fail_below = calculate_counts(filtered_tpa, q3_score_threshold, 'below')
                    global_q3_abs_data.append({'Category Threshold': category, 'Value': q3_score_threshold, 'Global HM Pass Count Above': pass_above, 'Global HM Fail Count Above': fail_above, 'Global HM Pass Count Below': pass_below, 'Global HM Fail Count Below': fail_below})

        # Display Percentage Tables
        st.subheader("Global Pass Rate Analysis Based on Median Score Thresholds (%)")
        if global_median_pct_data:
            st.dataframe(pd.DataFrame(global_median_pct_data))

        st.subheader("Global Pass Rate Analysis Based on Q3 Score Thresholds (%)")
        if global_q3_pct_data:
            st.dataframe(pd.DataFrame(global_q3_pct_data))

        # Display Absolute Count Tables
        st.subheader("Global Pass/Fail Counts Based on Median Score Thresholds")
        if global_median_abs_data:
            st.dataframe(pd.DataFrame(global_median_abs_data))

        st.subheader("Global Pass/Fail Counts Based on Q3 Score Thresholds")
        if global_q3_abs_data:
            st.dataframe(pd.DataFrame(global_q3_abs_data))

except FileNotFoundError:
    st.error("Error: 'tpa.csv' not found. Please make sure the file is in the same directory as the app.")
except Exception as e:
    st.error(f"An error occurred while processing the file: {e}")
