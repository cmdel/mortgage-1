from decimal import Decimal
import pytest
import logging

from mortgage import Loan, Builder, LoanFromSchedule


def convert(value):
    return Decimal(value).quantize(Decimal('0.01'))

def convert_rate(value):
    return Decimal(value).quantize(Decimal('1e-6'))



@pytest.fixture(scope='class')
def f2floan_350k() -> Loan:
    bldr = Builder()
    return bldr.build_fixed_to_floating_loan(intro_rate=.02, intro_term=2, svr=.05, principal=350_000, term=25)


@pytest.fixture(scope='class')
def f2floan_360k() -> Loan:
    bldr = Builder()
    return bldr.build_fixed_to_floating_loan(intro_rate=.0189, intro_term=2, svr=.0424, principal=360_000, term=20)

class TestFixedToFloatingLoan(object):
    logging.disable(logging.CRITICAL)
    
    def test_monthly_payment(self, f2floan_350k):
        assert convert(f2floan_350k.monthly_payment) == convert(1483.49)
    def test_original_balance(self, f2floan_350k):
        assert f2floan_350k.principal == convert(350000.00)

    # def test_interest_rate(self, f2floan_350k):
    #     assert f2floan_350k.interest == convert(.06)

    # def test_apy(self, f2floan_350k):
    #     assert f2floan_350k.apy == convert(6.17)

    # def test_apr(self, f2floan_350k):
    #     assert f2floan_350k.apr == convert(6.00)

    def test_term(self, f2floan_350k):
        assert f2floan_350k.term == 25

    def test_total_principal(self, f2floan_350k):
        assert f2floan_350k.total_principal == convert(350000.00)

    def test_total_interest(self, f2floan_350k):
        assert f2floan_350k.total_interest == convert(238153.43)

    def test_total_paid(self, f2floan_350k):
         assert f2floan_350k.total_paid == convert(588153.43)

    # def test_interest_to_principle(self, f2floan_350k):
    #     assert f2floan_350k.interest_to_principle == convert(115.8)

    def test_years_to_pay(self, f2floan_350k):
        assert f2floan_350k.years_to_pay == 25


    def test2_monthly_payment(self, f2floan_360k):
        assert convert(f2floan_360k.monthly_payment) == convert(1802.49)

    def test2_total_paid(self, f2floan_360k):
        assert f2floan_360k.total_paid == convert(515328.47)

    def test2_total_interest(self, f2floan_360k):
        assert f2floan_360k.total_interest == convert(155328.47)

    def test_aprc(self, f2floan_360k):
        assert f2floan_360k.aprc == convert_rate(.038)