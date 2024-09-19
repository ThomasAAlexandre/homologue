from tomologie import QuantumCircuit, QuantumRegister, state_tomography

"""
    Test 1 for state tomography 
    Expected value: 0.25 amplitude for all possible states
    Output value: [-2.49543734e-01-0.00000000e+00j -8.96642565e-04+2.49655645e-01j
 -1.76608838e-01+1.76804024e-01j  1.77283939e-01+1.75894252e-01j
  1.01665660e-04-2.49943123e-01j -2.50247031e-01-8.02927714e-05j
 -1.76670932e-01-1.76827321e-01j -1.77231718e-01+1.78172277e-01j
  2.49898223e-01-9.45811887e-04j -2.21232141e-04-2.50013259e-01j
  1.76712242e-01-1.77389663e-01j -1.77275329e-01-1.77168028e-01j
  7.34181653e-04+2.50342464e-01j  2.48997491e-01-4.03811364e-04j
  1.76008946e-01+1.77042632e-01j  1.76301559e-01-1.76932167e-01j]
"""
qreg_q = QuantumRegister(4, "q")
circuit = QuantumCircuit(qreg_q)
circuit.h(qreg_q[2])
circuit.h(qreg_q[3])
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.z(qreg_q[3])
circuit.s(qreg_q[2])
circuit.tdg(qreg_q[1])
circuit.sdg(qreg_q[0])

state = state_tomography(circuit, 100000)

print("Test 1:", state)

"""
    Test 2 for state tomography 
    Expected value: 0.707 amplitude for all possible states
    Output value: [-0.70805367+0.j        -0.70615837+0.0006073j]
"""
qreg_q = QuantumRegister(1, "q")
circuit_2 = QuantumCircuit(qreg_q)
circuit_2.h(qreg_q[0])
state = state_tomography(circuit_2, 100000)

print("Test 2:", state)
