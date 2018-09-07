"""This program runs a simple one-dimensional implementation of the 
quantum circuit learning algorithm as presented by 
Kosuke Mitarai, Makoto Negoro, Masahiro Kitagawa and Keisuke Fujii
in Quantum Circuit Learning (arXiv:1803.00745).

The numbers of steps (layers) and qubits are variables (nb_steps and nb_qubits).
I added a slight modification that a moderate amount of times edge cases are 
used to learned on (-1 or 1) in order to achieve smoother result at the edges 
of the domain. 

I implemented the network using a mean squared error objective function and 
utilizes online learning with stochastic gradient descent. Also it is written 
using numpy and the scipy matrix exponential instead of tensorflow, so it is 
very basic. The code has no output of learning details, but the network map
is visualized during learning."""
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Specify parameters
nb_steps = 6
nb_qubits = 6
episodes = 250
learning_rate = 0.15
edge_chance = 0.00
show_progress = True
plot_freq = 2

# Pauli matrices required for the model and the Ising propagator
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1.j],[1.j,0]])
sigma_z = np.array([[1,0],[0,-1]])

def Ising(nb_qubits):
	# Evolution as specified in the paper
	t = 10

	# Random magnetic field strengths for each site
	a = np.random.rand(nb_qubits)*2-1

	# Initialize local Hamiltonian
	H_0 = 0

	# Iterate over qubits
	for i in range(nb_qubits):
		# Prepare tensor product of identities
		eyes = [np.eye(2) for k in range(nb_qubits)]

		# Replace ith identity by first Pauli matrix
		eyes[i] = sigma_x

		# Calculate the appropriate Pauli group element
		tensor = 1
		for k in range(nb_qubits):
			tensor = np.kron(eyes[k],tensor)

		# Update Hamiltonian
		H_0 += a[i]*tensor

	# Random interaction strengths between all two-qubit combinations
	# (nb_qubits^2+nb_qubits)/2 of these are unused
	J = np.random.rand(nb_qubits,nb_qubits)*2-1

	# Initialize interaction Hamiltonian
	H_1 = 0

	# Iterate over all qubit pairs once
	for i in range(nb_qubits):
		for j in range(i+1, nb_qubits):
			# Prepare tensor product of identites
			eyes = [np.eye(2) for k in range(nb_qubits)]

			# Replace ith and jth identity by third Pauli matrix
			eyes[i] = sigma_z
			eyes[j] = sigma_z

			# Calculate the appropriate Pauli group element
			tensor = 1
			for k in range(nb_qubits):
				tensor = np.kron(eyes[k],tensor)

			# Update Hamiltonian
			H_1 += J[i][j]*tensor

	# Calculate total Hamiltonian
	H = H_0+H_1

	# Calculate propagator
	U = linalg.expm(-1.j*H*t)

	return U

def rho(x, nb_qubits):
	# Calculate single-qubit density matrix
	single_rho = np.eye(2)+x*sigma_x+np.sqrt(1-x**2)*sigma_z

	# Calculate nb_qubit-fold tensor product of density matrix
	tensor = 1
	for q in range(nb_qubits):
		tensor = np.kron(tensor, single_rho)

	# Return normalized densitry matrix
	return tensor/(2**nb_qubits)

class Network:
	def __init__(self, steps, learning_rate, nb_qubits):
		# Gather variables
		self.steps = steps
		self.learning_rate = learning_rate
		self.nb_qubits = nb_qubits

		# Initialize the Ising kernels for this model
		self.kernels = [Ising(self.nb_qubits) for s in range(self.steps)]

		# Initialize trainable weights
		self.weights = 2*np.pi*np.random.rand(self.steps, self.nb_qubits, 3)
		# Initialize the trainable scale
		self.scale = 1

		# Specify the output observable
		self.evaluator = np.kron(sigma_z,np.eye(2**(self.nb_qubits-1)))

		# Calculate initial circuit propagator
		self.U = self.propagator()

		# Prepare local group elements using the first and third Pauli matrix
		self.P_x = []
		self.P_z = []
		for q in range(nb_qubits):

			# Prepare tensor product of identities
			ones = [np.eye(2) for b in range(self.nb_qubits)]

			# Replace qth identity with first Pauli matrix
			ones[q] = sigma_x

			# Calculate Pauli group element
			P_x = 1
			for k in range(self.nb_qubits):
				P_x = np.kron(P_x,ones[k])

			# Replace (again) qth element with third Pauli matrix
			ones[q] = sigma_z

			# Calculate Pauli group element
			P_z = 1
			for k in range(self.nb_qubits):
				P_z = np.kron(P_z,ones[k])

			# Append Pauli group elements
			self.P_x.append(P_x)
			self.P_z.append(P_z)

	def singles_gates(self):
		# Prepare all single-qubit timesteps of the circuit
		timesteps = []

		# Iterate over amount of steps and qubits
		for s in range(self.steps):
			tensor = 1
			for q in range(self.nb_qubits):
				# Calculate matrix product of rotations making up the unitary
				u = linalg.expm(-1.j*sigma_x*self.weights[s,q,0])
				u = u.dot(linalg.expm(-1.j*sigma_z*self.weights[s,q,1]))
				u = u.dot(linalg.expm(-1.j*sigma_x*self.weights[s,q,2]))

				# Expand tensor product by new single-site unitary
				tensor = np.kron(tensor,u)
			# Append full tensor product to list of timesteps
			timesteps.append(tensor)

		return timesteps

	def propagator(self):
		# Get single-qubit timesteps
		singles = self.singles_gates()

		# Intialize full propagator as identity
		U = np.eye(2**self.nb_qubits)
		# Iterate over all steps
		for s in range(self.steps):
			# Evolve by the sth Ising kernel
			U = self.kernels[s].dot(U)
			# Evolve by the sth single-qubit timestep
			U = singles[s].dot(U)

		return U

	def train(self, x, y):
		# Gather single-qubit timesteps
		singles = self.singles_gates()

		# Initialize inner unitary for gradient
		U_inner = np.eye(2**self.nb_qubits)

		# Initialize gradient tensor as zeros
		gradients = np.zeros((self.steps, self.nb_qubits, 3))

		for s in range(self.steps):
			# Expand inner unitary by sth timestep and kernel
			U_inner = singles[s].dot(self.kernels[s]).dot(U_inner)
			# Calculate adjoint of inner unitary
			U_inner_t = U_inner.conj().transpose()

			# Calculate outer unitary for the sth timestep
			U_outer = np.eye(2**self.nb_qubits)
			for k in range(s+1,self.steps):
				U_outer = singles[k].dot(self.kernels[k]).dot(U_outer)
			# Calcuate adjoint of inner unitary
			U_outer_t = U_outer.conj().transpose()

			# Transform densitry matrix with inner unitary
			inner = U_inner.dot(x).dot(U_inner_t)

			# Calculate observable with outer unitary 
			outer = U_outer_t.dot(self.evaluator).dot(U_outer)

			for q in range(self.nb_qubits):
				# Rotations of the qth single-site unitary of the sth timestep
				R1 = linalg.expm(-1.j*self.P_x[q]*self.weights[s,q,0])
				R2 = linalg.expm(-1.j*self.P_z[q]*self.weights[s,q,1])
				R3 = linalg.expm(-1.j*self.P_x[q]*self.weights[s,q,2])

				# Calculate the corresponding adjoints
				R1_t = R1.conj().transpose()
				R2_t = R2.conj().transpose()
				R3_t = R3.conj().transpose()
				
				# Calculate the half rotations for the Pauli group elements
				self.Xplus = linalg.expm(-1.j*self.P_x[q]*np.pi/4)
				self.Zplus = linalg.expm(-1.j*self.P_z[q]*np.pi/4)

				# Calculate the corresponding adjoints
				self.Xminus = self.Xplus.conj().transpose()
				self.Zminus = self.Zplus.conj().transpose()

				# Calculate gradient for first rotation angle
				gradients[s,q,0] = 0.5*np.trace(
						 outer.dot(self.Xplus ).dot(inner).dot(self.Xminus)
						-outer.dot(self.Xminus).dot(inner).dot(self.Xplus )
					).real
				
				# Calculate gradient for second rotation angle
				gradients[s,q,1] = 0.5*np.trace(
						 outer.dot(R1).dot(self.Zplus ).dot(R1_t).dot(inner)
						 .dot(R1).dot(self.Zminus).dot(R1_t)
						-outer.dot(R1).dot(self.Zminus).dot(R1_t).dot(inner)
						.dot(R1).dot(self.Zplus ).dot(R1_t)
					).real

				# Calculate gradient for third rotation angle
				gradients[s,q,2] = 0.5*np.trace(
						 outer.dot(R1).dot(R2).dot(self.Xplus ).dot(R2_t).
						 dot(R1_t).dot(inner).dot(R1).dot(R2).dot(self.Xminus).
						 dot(R2_t).dot(R1_t)
						-outer.dot(R1).dot(R2).dot(self.Xminus).dot(R2_t).
						dot(R1_t).dot(inner).dot(R1).dot(R2).dot(self.Xplus ).
						dot(R2_t).dot(R1_t)
					).real

		# Calculate output density matrix
		z = self.evaluate(x)

		# Update weights and scale with hardcoded gradient of objective
		self.weights += self.learning_rate*2*(y-z)*gradients
		self.scale += self.learning_rate*2*(y-z)*z

		# Update circuit propagator
		self.U = self.propagator()

	def evaluate(self, x):
		# Transform input density matrix with circuit propagator
		y = self.U.dot(x).dot(self.U.conj().transpose())

		# Return scaled expectation value
		return np.trace(y.dot(self.evaluator)).real*self.scale

def target(x):
	# Get target from input

	return np.sin(x*np.pi)
	#return np.exp(x)
	#return x**2
	#return np.abs(x)

# Initialize network model
net = Network(nb_steps, learning_rate, nb_qubits)

# Generate domain sampling
domain =  np.linspace(-1,1,101)

# Generate data before training
initial = np.array([net.evaluate(rho(x, nb_qubits)) for x in domain])

# Initialize array for later teacher visualization
teacher_x = []

# Initial plot
fig = plt.gcf()
fig.show()
fig.canvas.draw()
inbetween = np.array([net.evaluate(rho(x, nb_qubits)) for x in domain])
plt.plot(domain,initial,
	linestyle='-.',color='green')
plt.plot(domain,inbetween,
	color='red')
plt.plot(domain,target(domain),
	linestyle=':',color='blue')
plt.xlim(-1, 1)
fig.canvas.draw()
plt.pause(0.01)

# Minibatch training over teacher data
for q in range(episodes):
	# Choose random input
	x = np.random.rand()*2-1

	# Moderate chance to replace input by -1 or 1 for better edges
	if np.random.rand()<edge_chance:
		x = np.random.choice(np.array([-1,1]))
	
	# Append input to teacher data
	teacher_x.append(x)
	
	# Calculate input states of minibatch
	rho_in = rho(x, nb_qubits)
	
	# Calculate targets of minibatch
	out_val = target(x)
	
	# Train on minibatch
	net.train(np.array(rho_in),np.array(out_val))
	
	# Plot intermediate results
	if show_progress and q%plot_freq == 0:
		inbetween = np.array([net.evaluate(rho(z, nb_qubits)) for z in domain])
		plt.clf()
		plt.plot(domain,initial,
			linestyle='-.',color='green')
		plt.plot(domain,inbetween,
			color='red')
		plt.plot(domain,target(domain),
			linestyle=':',color='blue')
		plt.plot(np.array(x),np.array(out_val), 
			linestyle='',color='blue', marker='.')
		plt.xlim(-1, 1)
		plt.draw()
		plt.pause(0.01)

# Generate data after training
after = np.array([net.evaluate(rho(x, nb_qubits)) for x in domain])

# Calculate targets of teacher data
teacher_y = target(np.array(teacher_x))

# Plot everything
plt.close()
plt.plot(domain,initial,
	linestyle='-.',color='green')
plt.plot(domain,after,
	color='red')
plt.plot(teacher_x,teacher_y,
	linestyle='',marker='.',color='blue',markersize=3)
plt.xlim(-1, 1)
plt.show()
