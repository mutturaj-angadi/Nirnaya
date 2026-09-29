from nirnaya_core import LPModel, Variable, Constraint, Sense, ObjectiveSense

INF = float("inf")


def var(name, lower=0.0, upper=INF, is_integer=False):
    return Variable(name=name, lower=lower, upper=upper, is_integer=is_integer)


def con(name, coeffs, sense, rhs):
    return Constraint(name=name, coefficients=coeffs, sense=sense, rhs=rhs)


def model(variables, constraints, objective, sense=ObjectiveSense.MINIMIZE, offset=0.0, name="m"):
    return LPModel(
        variables=variables,
        constraints=constraints,
        objective=objective,
        objective_sense=sense,
        objective_offset=offset,
        name=name,
    )
