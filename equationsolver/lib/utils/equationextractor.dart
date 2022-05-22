class EquationExtractor {
  int getEquationDegree(String equation) {
    int highestNumber = 1;
    for (int i = 0; i < equation.length - 1; i++) {
      if (RegExp(r'^[a-z]').hasMatch(equation[i]) &&
          RegExp(r'^[0-9]').hasMatch(equation[i + 1])) {
        int newNumber = int.parse(equation[i + 1]);
        if (newNumber > highestNumber) {
          highestNumber = newNumber;
        }
      }
    }
    return highestNumber;
  }
}
