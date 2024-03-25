"""
Generate insights from Intuit Mint transaction data.
"""

import collections
import re
from typing import Dict, List, Optional, Union

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

StrFilter = Optional[List[str]]


class TX:
    """Transaction"""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df["TX_Date"] = pd.to_datetime(self.df["Date"])
        self._ensure_types()

    def _ensure_types(self):
        str_fields = [
            "Description",
            "Original Description",
            "Category",
            "Transaction Type",
            "Account Name",
            "Account",
            "Original Statement",
        ]
        for field in str_fields:
            if field in self.df.columns:
                self.df[field] = self.df[field].astype(str)

    def head(self, n: Optional[int] = None):
        print(self.df.shape)
        print(self.df.columns)
        if n is not None:
            print(self.df.head(n))
        else:
            print(self.df.head())

    @classmethod
    def clean_statement(cls, statement: str) -> str:
        txt = statement.replace("\n", " ")
        txt = " ".join(txt.split())
        txt = statement.strip()
        return txt

    @classmethod
    def prep_statement_equality(cls, statement: str) -> str:
        txt = re.sub(r"[^a-zA-Z0-9\s\-\.]", "", statement)
        txt = txt.replace("-", "")
        txt = cls.clean_statement(txt).lower()
        txt = re.sub(r"\s+", " ", txt)
        return txt

    @classmethod
    def statements_equal(cls, statement: str, other: str) -> bool:
        return cls.prep_statement_equality(statement) == cls.prep_statement_equality(other)

    def export(self, path: str):
        # drop added rows starting with TX_
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^TX_")]
        self.df.to_csv(path, index=False)
        print(f"Exported to {path}")

    def list_categories(self):
        """categories and count"""
        counts = collections.Counter(self.df["Category"])
        # print(f"Unique categories: {len(counts)}")
        # print("most common:", counts.most_common(10))
        # print("least common:", counts.most_common()[-10:])
        return counts
        print([cat for cat, _ in sorted(mint_cats.items(), key=lambda x: x[1], reverse=True)])

    def accounts(self):
        col_name = "Account Name"
        if not col_name in self.df.columns:
            col_name = "Account"
        counts = collections.Counter(self.df[col_name])
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)


class MonarchTransactions(TX):
    """
    Date,Merchant,Category,Account,Original Statement,Notes,Amount,Tags
    2024-02-26,Payments Money Out ,Transfer,**********8968 (...8968),M1-Payme,,-200.00,
    2024-02-26,Chase,Credit Card Payment,**,Chase Credit Crd-Autopay,,-2934.30,
    2024-02-25,Veracruz,Restaurant,CREDIT C,VERACRUZ - 0,,-15.48,
    """

    def __init__(self, csv_path: str):
        super().__init__(csv_path)


class MintTransations(TX):
    """
    input sample:
    "Date","Description","Original Description","Amount","Transaction Type","Category","Account Name"
    "7/13/2023","Deposit of ","Deposit of $350.00 ,"350.00","credit","Transfer","INDIVIDUAL"
    "7/13/2023","Dividend","Dividend","2.61","credit","Investments","MyCl","",""
    "7/12/2023","AMZN Mktp U","AMZN Mktp US      ","13.13","credit","Shopping","Customized ","",""
    "7/12/2023","NX WASH,HUNT","NX ","15.00","debit","Service & Parts","Visa Signature"
    """

    def __init__(self, csv_path):
        super().__init__(csv_path)
        # self.excluded_categories = ["transfer", "credit card payment"]
        self.excluded_categories = []
        self.excluded_description = []
        self.df["TX_Amount"] = self.df["Amount"].where(
            self.df["Transaction Type"] == "credit", -self.df["Amount"]
        )
        # apply clean_statement
        self.df["Original Description"] = self.df["Original Description"].apply(TX.clean_statement)
        # self.df = self.filter_transactions(
        #     descriptions=self.excluded_description,
        #     categories=self.excluded_categories,
        #     exclude=True,
        # )

    def filter_transactions(
        self,
        descriptions: StrFilter = None,
        categories: StrFilter = None,
        start_date: StrFilter = None,
        end_date: StrFilter = None,
        exclude: bool = False,
    ):
        df = self.df
        for txt_filter in descriptions or []:
            evaluation = df["Description"].str.lower().str.contains(txt_filter.lower())
            if exclude:
                evaluation = ~evaluation
            df = df[evaluation]

        for txt_filter in categories or []:
            evaluation = df["Category"].str.lower().str.contains(txt_filter.lower())
            if exclude:
                evaluation = ~evaluation
            df = df[evaluation]

        return df

    @property
    def total_inbound(self):
        return self.df[self.df["Transaction Type"] == "credit"]["Amount"].sum()

    @property
    def total_outbound(self):
        return self.df[self.df["Transaction Type"] == "debit"]["Amount"].sum()

    def monthly(self, window_size):
        """
        Create a monthly DF that calculates moving average of income and expense.
        In the end we want the df to have: 'month', 'income', 'expense',
        'cash_flow', 'transaction_count'
        """
        df = self.df
        df["month"] = df["TX_Date"].dt.to_period("M")
        df["income"] = df["Amount"].where(df["Transaction Type"] == "credit", np.nan)
        df["expense"] = df["Amount"].where(df["Transaction Type"] == "debit", np.nan)
        df["cash_flow"] = df["Amount"].where(df["Transaction Type"] == "credit", -df["Amount"])
        df["transaction_count"] = 1

        df = df.groupby(["month"]).agg(
            {
                "income": "sum",
                "expense": "sum",
                "cash_flow": "sum",
                "transaction_count": "sum",
            }
        )

        df["income_moving_avg"] = df["income"].rolling(window_size).mean()
        df["expense_moving_avg"] = df["expense"].rolling(window_size).mean()
        df["cash_flow_moving_avg"] = df["cash_flow"].rolling(window_size).mean()

        return df

    def to_dict(self):
        return {
            "total_inbound": self.total_inbound,
            "total_outbound": self.total_outbound,
            "cash_flow": self.total_inbound - self.total_outbound,
        }

    def save_trend(self, window_size=3):
        """
        generate a line chart that shows infrom from the monthly df
        """
        df = self.monthly(window_size)

        df = df.rolling(window_size).mean()

        # trim the part the data is empty
        df = df[df["income_moving_avg"].notnull()]

        # createa trendline for cashflow
        df["cash_flow_trend"] = np.poly1d(
            np.polyfit(
                df.index.astype(np.int64) / 10**9,
                df["cash_flow_moving_avg"],
                1,
            )
        )(df.index.astype(np.int64) / 10**9)

        ax = df.plot.line(
            y=[
                "income_moving_avg",
                "expense_moving_avg",
                "cash_flow_moving_avg",
                "cash_flow_trend",
            ],
            ylabel="USD",
            title="Monthly cash flow",
            figsize=(20, 10),
        )
        # draw labels and grid lines
        ax.set_xticklabels(df.index.strftime("%Y-%m"))
        ax.grid(True, which="both")
        ax.axhline(y=0, color="k")

        plt.savefig("monthly_cash_flow.png")

    def print_report(self, window_size=3):
        print(self.df)
        print(self.monthly(window_size))
        # print(render.human_readable_dict(self.to_dict()))
        # find all transactions that have original description set
        self.head()

    def print_differing_descriptions(self):
        df = self.df[self.df["Description"] != self.df["Original Description"]]
        slim = df[["Description", "Original Description"]]
        print(slim.to_string(index=False))

    def remove_duplicates(self):
        # TODO: fixme is this safe?
        pre_rows = len(self.df)
        df_unique = self.df.drop_duplicates(
            subset=["Date", "Amount", "Account Name", "Transaction Type", "Category"]
        )
        print(f"Removed {pre_rows - len(df_unique)} duplicates out of {pre_rows} rows.")
        self.df = df_unique

    def remove_duplicates_v2(self):
        """mint exports can have duplicates
        based on https://github.com/bulutmf/mint2monarch
        """

        def clean(df: pd.DataFrame, d: pd.Timestamp):
            # filtering transactions by date and resetting index
            df_all_trans = df[df["Date"] == d].reset_index(drop=True)
            df_all_trans["idx"] = df_all_trans.index.tolist()
            new_list = []
            ignore_idxs = []
            for idx, row in df_all_trans.iterrows():
                if idx in ignore_idxs:
                    continue
                new_list.append(row.to_dict())
                amount = row["Amount"]
                account_name = row["Account Name"]
                trans_type = row["Transaction Type"]
                # category = row["Category"]
                # finding duplicates based on specific criteria
                df_same_amounts = df_all_trans[
                    (df_all_trans["Amount"] == amount)
                    & (df_all_trans["Account Name"] == account_name)
                    & (df_all_trans["Transaction Type"] == trans_type)
                    & (df_all_trans["idx"] != idx)
                ]
                indices = df_same_amounts.index.tolist()
                ignore_idxs.extend(indices)
            ignored_list = df_all_trans[df_all_trans["idx"].isin(ignore_idxs)].to_dict(
                orient="records"
            )
            return new_list, ignored_list

        unique_dates = self.df["Date"].unique().tolist()
        all_unique_trans = []
        all_ignored_trans = []
        for d in unique_dates:
            transactions, ignored_trans = clean(self.df, d)
            all_unique_trans.extend(transactions)
            all_ignored_trans.extend(ignored_trans)
        # for ig in all_ignored_trans:
        #     print(ig)

        # compiling unique transactions
        df_new = pd.DataFrame(all_unique_trans)
        del df_new["idx"]
        print(f"Removed {len(self.df) - len(df_new)} duplicates out of {len(self.df)} rows.")
        self.df = df_new


def statements_match(mint: str, monarch: str) -> bool:
    """
    compare min and monarch statements
    has assumptions about how monarch and mint statements are created.
    """
    monarch = TX.prep_statement_equality(monarch)
    mint = TX.prep_statement_equality(mint)

    # monarch simplifications
    known_monarch_statements: Dict[str, Union[str, List[str]]] = {
        "amazon": "amz",
        "7-eleven": "7 eleven",
        "whole foods": "wholefds",
        "trader joes": "trader joe",
        "american airlines": "american",
        "google cloud storage": "google storage",
        "in-n-out": "in n out",
        "walmart": ["wal-mart", "wm"],
    }
    # if these words appear in both they're the same
    magic_words = {"vanguard", "m1-payments", "openai", "pueblo", "booking.com", "bolt.eu"}
    if monarch == mint:
        return True
    if monarch in mint:
        # monarch shows stripped down versions of the original statement
        return True
    if mint in monarch:
        return True
    if monarch in known_monarch_statements:
        checks = known_monarch_statements[monarch]
        if isinstance(checks, str):
            checks = [checks]
        for check in checks:
            if check in mint:
                return True
    for word in magic_words:  # PERF
        if word in monarch and word in mint:
            return True
    # FIXME: it's a clutch to not cleaning the spaces correctly I think
    s1 = re.sub(r"\s", "", monarch)
    s2 = re.sub(r"\s", "", mint)
    if s1 == s2:
        return True

    # remove digits and spaces
    s1 = re.sub(r"\d", "", s1)
    s2 = re.sub(r"\d", "", s2)
    if s1 == s2:
        return True

    # take the first word from monarch and see if in mint
    s1 = monarch.split(" ")[0].replace(".", "")
    if s1 in mint.replace(".", ""):
        return True

    return False


def remove_existing(mint: MintTransations, monarch: MonarchTransactions):
    """finds overlaps. returns a df with overlapping rows."""
    # write a comparison function based on date, and amount
    # "Original Description" from mint and "Original Statement" from Monarch
    # plus amounts considering the signs.
    matches = []  # Initialize an empty list for matching rows
    partial_matches = 0
    monarch.df["Original Statement"] = monarch.df["Original Statement"].apply(
        TX.prep_statement_equality
    )
    # Iterate over Mint transactions
    for index, mint_row in mint.df.iterrows():
        mint_desc = mint_row["Original Description"]
        mint_desc = TX.prep_statement_equality(mint_desc)
        # Filter Monarch transactions for matches
        monarch_matched_rows = monarch.df[
            (monarch.df["TX_Date"] == mint_row["TX_Date"])
            & (monarch.df["Amount"].astype(float) == float(mint_row["TX_Amount"]))
            # & (monarch.df['Original Statement'] == mint_desc)
        ]

        if monarch_matched_rows.empty:
            continue

        is_in_monarch = False
        for _, matched_row in monarch_matched_rows.iterrows():
            monarch_desc = matched_row["Original Statement"]
            monarch_desc = TX.prep_statement_equality(monarch_desc)
            if statements_match(mint_desc, monarch_desc):
                is_in_monarch = True
                break
            else:
                amount = mint_row["TX_Amount"]
                date = mint_row["Date"]
                if abs(float(amount)) >= 150:  # TODO: parametrize
                    print(f"### Skipped partial match for mint row: {date} {amount}")
                    print(f"MINT:{mint_desc}")
                    print(f"MONARCH:{monarch_desc}")

        if is_in_monarch:
            matches.append(
                {
                    "Date": mint_row["Date"],
                    "Amount": mint_row["TX_Amount"],
                    "MintDescription": mint_row["Original Description"],
                }
            )
            mint.df = mint.df.drop(index=index)
        else:
            partial_matches += 1

    # Convert the list of matches to a DataFrame and return
    print(f"allowed in {partial_matches} partial duplicates/matches")
    return pd.DataFrame(matches)


# keys in mint, destination in monarch
category_mapping = {
    "Food & Dining": "Restaurants & Bars",
    "Investments": "Investment",
    "Transfer": "Transfer",
    "Shopping": "Shopping",
    "Auto & Transport": "Taxi & Ride Shares",  # Consider 'Auto Payment' or 'Gas' if it fits better.
    "Travel & Vacation": "Travel & Vacation",
    "Groceries": "Groceries",
    "Income": "Other Income",  # If it aligns more with regular income; otherwise, 'Other Income'.
    "Uncategorized": "Uncategorized",
    "Entertainment": "Entertainment & Recreation",
    "Bills & Utilities": "Other Utilities",
    "Business Services": "Business Utilities & Communication",
    "Mortgage & Rent": "Rent",  # 'Rent (Short Term)' could apply if it's more about temporary lodging.
    "Taxes": "Taxes",
    "Fees & Charges": "Financial Fees",  # Could also consider 'Parking & Tolls' if related to transport.
    "Medical": "Medical",
    "Home & Garden": "Home Improvement",  # 'Garden' aspect is assumed under home improvements.
    "Personal Care": "Personal",  # Could consider 'Fitness' if it involves health club memberships or similar.
    "Education": "Miscellaneous",  # No direct match; this remains a catch-all.
    "Cash & Checks": "Cash & ATM",  # Straightforward mapping.
    "Gifts": "Gifts",
    "Donations": "Gifts",
}


def merge_mint_monarch(mint_csv: str, monarch_csv: str, output="deduped_mint.csv"):
    mint = MintTransations(mint_csv)
    monarch = MonarchTransactions(monarch_csv)

    mint.remove_duplicates()
    mint.remove_duplicates_v2()
    duplicates = remove_existing(mint, monarch)
    print(f"{len(duplicates)} rows of duplicates.")
    print("mint transactions post deduplication against monarch:", mint.df.shape)

    mint_cats = mint.list_categories()
    mo_cats = monarch.list_categories()
    # print("mint categories")
    # print([cat for cat, _ in sorted(mint_cats.items(), key=lambda x: x[1], reverse=True)])
    # print("monarch categories")
    # print([cat for cat, _ in sorted(mo_cats.items(), key=lambda x: x[1], reverse=True)])

    for key in category_mapping.keys():
        assert key in mint_cats, f"category {key} not found in mint"
    for val in category_mapping.values():
        assert val in mo_cats, f"category {val} not found in monarch"

    def map_mint_category(mint_cat: str):
        if mint_cat in mo_cats:
            # category exists in Monarch
            return mint_cat
        if mint_cat in category_mapping:
            return category_mapping[mint_cat]
        return "Uncategorized"

    mint.df["Category"] = mint.df["Category"].apply(map_mint_category)

    mint.export(output)


def test(mint_csv: str, monarch_csv: str, arg: str = ""):
    mint = MintTransations(mint_csv)
    monarch = MonarchTransactions(monarch_csv)

    return mint.accounts()


if __name__ == "__main__":
    fire.Fire(
        {
            "mint": MintTransations,
            "monarch": MonarchTransactions,
            "merge": merge_mint_monarch,
            "test": test,
        }
    )

