"""
mortgage.loan
~~~~~~~~~~~~~~~~

This module provides a Loan object to create and calculate various mortgage statistics.
"""

import logging

from os import path
from collections import namedtuple
from decimal import Decimal
from typing import Tuple, List, Any
from logging.config import fileConfig
from scipy import optimize


Installment = namedtuple('Installment', 'number payment interest principal total_interest balance')

# Setup the logger
log_file_path = path.join(path.dirname(path.abspath(__file__)), '../logging_config.ini')
fileConfig(log_file_path)
logger = logging.getLogger(__name__)

class Loan():
    """A user-created :class:`Loan <Loan>` object for creating a loan, calculating amortization
    schedule, and showing statistics.

    :param principal: The original sum of money borrowed.
    :param interest: The amount charged by lender for use of the assets.
    :param term: The lifespan of the loan.
    :param term_unit: Unit for the lifespan of the loan.
    :param compounded: Frequency that interest is compounded

    Usage:
        >>> from mortgage import Loan
        >>> Loan(principal=200000, interest=.04125, term=15)
        <Loan principal=200000, interest=0.04125, term=15>
    """
    def __init__(self, principal: int, interest: float, term: int, term_unit: str = 'years', compounded:str = 'monthly', initial_interest_amount: Decimal = 0.0) -> None:

        term_units = {'days', 'months', 'years'}
        compound = {'daily', 'monthly', 'annually'}

        assert principal > 0, 'Principal must be positive value'
        assert 0 <= interest <= 1, 'Interest rate must be between zero and one'
        assert term > 0, 'Term must be a positive number'
        assert term_unit in term_units, 'term_unit can be either  days, months, or years'
        assert compounded in compound, 'Compounding can occur daily, monthly, or annually'

        periods = {
            'daily': 365.25,
            'monthly': 12,
            'annually': 1
        }

        self.principal = Decimal(principal)
        self.interest = Decimal(interest * 100) / 100
        self.term = term
        self.term_unit = term_unit
        self.compounded = compounded
        self.n_periods = periods[compounded]
        self._schedule = self._amortize(initial_interest_amount)
        logger.debug("Created loan with principal=%s, interest={self.interest}, term={self.term}",self.principal)

    def __repr__(self) -> str:
        return f'<Loan principal={self.principal}, interest={self.interest}, term={self.term}>'

    @staticmethod
    def _quantize(value: Any) -> Decimal:
        return Decimal(value).quantize(Decimal('0.01'))

    @staticmethod
    def _quantize_rate(value: Any) -> Decimal:
        return Decimal(value).quantize(Decimal('1e-6'))

    def schedule(self, nth_payment: Any=None) -> Any:
        """Retreive payment information for the nth payment.

                Usage:
                    >>> from mortgage import Loan
                    >>> loan = Loan(principal=200000, interest=.06, term=30)
                    >>> loan.schedule(1)
                    Installment(number=1, payment=Decimal('1199.101050305504789182922487'), interest=Decimal('1E+3'), principal=Decimal('199.101050305504789182922487'), total_interest=Decimal('1000'), balance=Decimal('199800.8989496944952108170775'))
                """
        if nth_payment:
            data = self._schedule[nth_payment]
        else:
            data = self._schedule
        return data

    @property
    def _monthly_payment(self):
        principal = self.principal
        _int = self.interest
        num = self.n_periods
        term = self.term
        payment = principal * _int / num / (1 - (1 + _int / num) ** (- num * term))
        return payment

    @property
    def monthly_payment(self) -> Decimal:
        """The total monthly payment (principal and interest) for the loan.

        Usage:
            >>> from mortgage import Loan
            >>> loan = Loan(principal=200000, interest=.06, term=30)
            >>> loan.monthly_payment
            Decimal('1199.10')
        """
        return self._quantize(self._monthly_payment)

    def _simple_interest(self, term):
        amt = self.principal * self.interest * term
        return self._quantize(amt)

    @property
    def apr(self) -> Decimal:
        """The annual percentage rate (or APR) is the amount of interest on your total loan amount
        that you'll pay annually (averaged over the full term of the loan)

        Usage:
            >>> from mortgage import Loan
            >>> loan = Loan(principal=200000, interest=.06, term=15)
            >>> loan.apr
            Decimal('6.00')
        """
        new_payment = self._simple_interest(term=1)
        apr = new_payment / self.principal
        return self._quantize(apr * 100)

    @property
    def aprc(self) -> Decimal:
        """This is the EU regulation designated 
        Annual Percentage Rate of Change.
        More info here: https://en.wikipedia.org/wiki/Annual_percentage_rate#European_Union
        """
        def f(x):
            return (float(self._monthly_payment) * ( (1 - 1 / (1 + x)**self.term) / ((1 + x)**(1 / self.n_periods) - 1)) - float(self.principal))
        root = optimize.newton(f, float(self.interest), tol=1e-8)
        return self._quantize_rate(root)

    @property
    def apy(self) -> Decimal:
        """The annual percentage yield (APY) is the effective annual rate of return taking into
        account the effect of compounding interest.

        Usage:
            >>> from mortgage import Loan
            >>> loan = Loan(principal=200000, interest=.06, term=15)
            >>> loan.apy
            Decimal('6.17')
        """
        apy = (1 + self.interest / self.n_periods) ** self.n_periods - 1
        return self._quantize(apy * 100)

    @property
    def ear(self) -> Decimal:
        """The effective annual rate (EAR) of return taking into
        account the effect of compounding interest. This the same 
        as APR.

        Usage:
            >>> from mortgage import Loan
            >>> loan = Loan(principal=200000, interest=.06, term=15)
            >>> loan.ear
            Decimal('6.17')
        """
        return self.apy

    @property
    def total_principal(self) -> Decimal:
        """Total principal paid over the life of the loan.

        Usage:
            >>> from mortgage import Loan
            >>> loan = Loan(principal=200000, interest=.06, term=15)
            >>> loan.total_principal
            Decimal('200000.00')
        """
        return self._quantize(self.principal)

    @property
    def total_interest(self) -> Decimal:
        """Total interest paid over the life of the loan.

        Usage:
            >>> from mortgage import Loan
            >>> loan = Loan(principal=200000, interest=.06, term=15)
            >>> loan.total_interest
            Decimal('103788.46')
        """
        return self._quantize(self.schedule(self.term * 12).total_interest)

    @property
    def total_paid(self) -> Decimal:
        """Total paid (principal and interest) over the life of the loan.

        Usage:
            >>> from mortgage import Loan
            >>> loan = Loan(principal=200000, interest=.06, term=15)
            >>> loan.total_paid
            Decimal('303788.46')
        """
        return self.total_principal + self.total_interest

    @property
    def interest_to_principle(self) -> Decimal:
        """Property that returns percentage of the principal is payed 
        to the bank over the life of the loan in interest charges.

        Usage:
            >>> from mortgage import Loan
            >>> loan = Loan(principal=200000, interest=.06, term=15)
            >>> loan.interest_to_principle
            Decimal('51.9')
        """
        return round(self.total_interest / self.total_principal * 100, 1)

    @property
    def years_to_pay(self) -> float:
        """Property that returns how many years it will take to pay off this loan given the
        payment schedule.

        Usage:
            >>> from mortgage import Loan
            >>> loan = Loan(principal=200000, interest=.06, term=15)
            >>> loan.years_to_pay
            15.0
        """
        return round(self.term * self.n_periods / 12, 1)

    @property
    def summarize(self):
        """Print out summary statistics for the loan structure
        """
        print('Original Balance:                        £{:>11,}'.format(self.principal))
        print('Interest Rate:                            {:>11} '.format(self.interest))
        print('APY:                                      {:>11} %'.format(self.apy))
        print('APR:                                      {:>11} %'.format(self.apr))
        print('Term:                                     {:>11} {}'.format(self.term, self.term_unit))
        print('Monthly Payment:                         £{:>11}'.format(self._quantize(self.monthly_payment)))
        print('Interest Only Monthly Payment:           £{:>11,}'.format(self._quantize(self.split_payment(1, self.monthly_payment)[0])))
        print('')
        print('Total principal payments:                £{:>11,}'.format(self.total_principal))
        print('Total interest payments:                 £{:>11,}'.format(self.total_interest))
        print('Total payments:                          £{:>11,}'.format(self.total_paid))
        print('')
        print('Average Capital Loan Annual Payments:    £{:>11,}'.format(self.total_paid / Decimal(self.years_to_pay)))
        print('Average Interest Only Annual Payments:   £{:>11,}'.format(self.total_interest / Decimal(self.years_to_pay)))        
        print('')
        print('Interest to principal:                   {:>11} %'.format(self.interest_to_principle))
        print('Years to pay:                            {:>11}'.format(self.years_to_pay))

    def split_payment(self, number: int, amount: Decimal) -> Tuple[Decimal, Decimal]:
        """Splits payment amount into principal and interest.

        :param number: the payment number (e.g. nth payment)
        :param amount: the total payment amount to be split

        Usage:
            >>> from mortgage import Loan
            >>> from decimal import Decimal
            >>> loan = Loan(principal=200000, interest=.06, term=15)
            >>> loan.split_payment(number=180, amount=Decimal(1199.10))
            (Decimal('8.396585353715933437157525763'), Decimal('1190.703414646283975613372297'))
        """

        def compute_interest_portion(payment_number):
            _int = self.interest / 12
            _intp1 = _int + 1

            numerator = self.principal * _int * (_intp1 ** (self.n_periods * self.term + 1)
                                                 - _intp1 ** payment_number)
            denominator = _intp1 * (_intp1 ** (self.n_periods * self.term) - 1)
            return numerator / denominator

        interest_payment = compute_interest_portion(number)
        principal_payment = amount - interest_payment
        return interest_payment, principal_payment

    def _amortize(self, initial_interest_amount: Decimal) -> List[Installment]:
        initialize = Installment(number=0,
                                 payment=0,
                                 interest=0,
                                 principal=0,
                                 total_interest=initial_interest_amount,
                                 balance=self.principal)
        schedule = [initialize]
        total_interest = initial_interest_amount
        balance = self.principal
        for payment_number in range(1, self.term * self.n_periods + 1):

            split = self.split_payment(payment_number, self._monthly_payment)
            interest_payment, principal_payment = split

            total_interest = total_interest + float(interest_payment)

            balance -= principal_payment
            installment = Installment(number=payment_number,
                                      payment=self._monthly_payment,
                                      interest=interest_payment,
                                      principal=principal_payment,
                                      total_interest=total_interest,
                                      balance=balance)

            schedule.append(installment)

        return schedule

class LoanFromSchedule(Loan):
    
    def __init__(self, schedule, principal: int, interest: float, term: int, step_payment: float, intro_term: int, term_unit: str = 'years', compounded: str = 'monthly'):
        term_units = {'days', 'months', 'years'}
        compound = {'daily', 'monthly', 'annually'}

        assert principal > 0, 'Principal must be positive value'
        assert 0 <= interest <= 1, 'Interest rate must be between zero and one'
        assert term > 0, 'Term must be a positive number'
        assert term_unit in term_units, 'term_unit can be either  days, months, or years'
        assert compounded in compound, 'Compounding can occur daily, monthly, or annually'

        super().__init__(principal, interest, term, term_unit, compounded)
        self._intro_term = intro_term
        self._step_payment = step_payment
        self._schedule = schedule

    def __repr__(self) -> str:
        return f'<Loan schedule={self._schedule}, principal={self.principal}, term={self.term}>'

    @property
    def monthly_payment(self) -> Tuple[Decimal]:
        return super().monthly_payment

    @property
    def step_payment(self) -> float:
        return self._quantize(self._step_payment)

    @property
    def aprc(self) -> Decimal:
        """This is the EU regulation designated 
        Annual Percentage Rate of Change.
        More info here: https://en.wikipedia.org/wiki/Annual_percentage_rate#European_Union
        """
        def f(x):
            return (float(self._monthly_payment) * ( (1 - 1 / (1 + x)**self._intro_term) / ((1 + x)**(1 / self.n_periods) - 1)) +
                    float(self._second_payment) * ( (1 - 1 / (1 + x)**(self.term-self._intro_term)) / ((1 + x)**(1 / self.n_periods) - 1)) - float(self.principal))

        root = optimize.newton(f, float(self.interest), tol=1e-8)
        return self._quantize_rate(root)

    @property
    def summarize(self):
        """Print out summary statistics for the loan structure
        """
        print('Original Balance:                        £{:>11,}'.format(self.principal))
        print('Interest Rate:                            {:>11} '.format(self._quantize(self.interest)))
        print('APRC:                                     {:>11} %'.format(self.aprc))
        print('APR:                                      {:>11} %'.format(self.apr))
        print('Term:                                     {:>11} {}'.format(self.term, self.term_unit))
        print('Monthly Payment:                         £{:>11}'.format(self._quantize (self.monthly_payment)))
        print('Interest Only Monthly Payment:           £{:>11,}'.format(self._quantize(self.split_payment(1, self.monthly_payment)[0])))
        print('')
        print('Total principal payments:                £{:>11,}'.format(self.total_principal))
        print('Total interest payments:                 £{:>11,}'.format(self.total_interest))
        print('Total payments:                          £{:>11,}'.format(self.total_paid))
        print('')
        print('Year 1 Total Cost:                       £{:>11,}'.format(self.principal - self._quantize(self._schedule[12][5]) + self._quantize(self._schedule[12][4])))
        print('Average Capital Loan Annual Payments:    £{:>11,}'.format(self._quantize(self.total_paid / Decimal(self.years_to_pay))))
        print('Average Interest Only Annual Payments:   £{:>11,}'.format(self._quantize(self.total_interest / Decimal(self.years_to_pay))))        
        print('')
        print('Interest to principal:                   {:>11} %'.format(self.interest_to_principle))
        print('Years to pay:                            {:>11}'.format(self.years_to_pay))


class Builder():
    """Loan consisting of an introductory fixed term, fixed rate
    and a subsequent remainder term with Standard Variable Rate (SVR)
    until maturity.
    """
    def __init__(self):
        pass

       
    def build_fixed_to_floating_loan(self, intro_rate, intro_term: int, svr, principal, term) -> Loan:
        assert 0 <= intro_rate <= 1, 'Interest rate must be between zero and one'
        assert 0 <= svr <= 1, 'Interest rate must be between zero and one'
        assert intro_term > 0, 'Term must be a positive number'

        self.intro_rate = Decimal(intro_rate * 100) / 100
        self.intro_term = intro_term

        self.svr = Decimal(svr * 100) / 100
        self.svr_term = term - intro_term

        self.principal = principal
        self.term = term

        intro_term_installments = self.intro_term * 12
        intro_loan = Loan(self.principal, self.intro_rate, self.term)
        # logger.debug("Created intro_loan: %s", intro_term_installments)
        comp_schedule = []
        for index, payment in enumerate(intro_loan.schedule()):
            comp_schedule.append(payment)
            if index == intro_term_installments: break # only take installments within the intro_term
        
        carried_fwd_total_interest = intro_loan.schedule(self.intro_term*12)[4]
        srv_loan = Loan(comp_schedule[-1][5], self.svr, self.svr_term, initial_interest_amount=carried_fwd_total_interest)
        logger.debug("Created loan: %s", srv_loan)
        logger.debug("Term time: %s", self.svr_term)
        srv_loan_sched = srv_loan.schedule()
        logger.debug("Installments No: {0}".format(len(srv_loan_sched)))
        #change the number of installment to match the final payment on the intro_term
        intro_term_installments = intro_term_installments + 1
        for i in range(len(srv_loan_sched)):
            srv_loan_sched[i] = srv_loan_sched[i]._replace(number = i + intro_term_installments)
            
        for index, payment in enumerate(srv_loan_sched[1:]):
            comp_schedule.append(payment)
        #TODO pass in the intro_term_type to be able to supply months not just years
        return LoanFromSchedule(comp_schedule, self.principal, self.intro_rate, self.term, srv_loan_sched[1][1], self.intro_term)