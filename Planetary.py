import numpy as np

#### Define useful constants

# All units are AU, days, and solar masses unless otherwise specified.
G = 0.01720209895**2
obl_earth = 23.439291111111*np.pi/180 # Earth obliquity
# Constants for magnitudes:
A1 = 3.33
B1 = 0.63
A2 = 1.87
B2 = 1.22
G_phot = 0.15


#### Function and class definitions

class Orbit:
	"""
	Define an orbit given the gravitational parameter and five fixed elements.
	Required initialization inputs:
		a -- semi-major axis, in AU
		e -- eccentricity
		i -- inclination, in radians
		anode -- longitude of ascending node, in radians
		argper -- argument of periapsis, in radians
		mu -- gravitational parameter G*(M+m), in AU^3 day^-2
	Optional initialization input:
		deg -- if True, all initialization angles are interpreted as degrees. Default is False.
		epoch -- Julian Date giving epoch
		M_epoch -- Mean anomaly at epoch, in radians
		name -- Name of orbiting body
	Additional attributes after initialization:
		T -- orbital period, in days
		TJ -- Jupiter Tisserand parameter
		M -- M+m, mass of orbiting body plus host
	"""
	def __init__(self, a, e, i, anode, argper, mu=G, epoch=None, M_epoch=None, deg=False, name=None):
		self.a = a
		self.e = e
		self.i = i*np.pi/180 if deg else i
		self.anode = anode*np.pi/180 if deg else anode
		self.argper = argper*np.pi/180 if deg else argper
		self.mu = mu
		self.T = 2*np.pi*np.sqrt((self.a**3)/self.mu)
		self.TJ = (5.20336301/self.a) + 2*np.cos(self.i)*np.sqrt((1 - self.e**2)*self.a/5.20336301)
		if (epoch is not None) and (M_epoch is not None):
			self.epoch = epoch
			self.M_epoch = M_epoch*np.pi/180 if deg else M_epoch
		elif (epoch is None) ^ (M_epoch is None):
			existing = 'epoch' if (epoch is not None) else 'M_epoch'
			print('Warning in Orbit.__init__(): must have both epoch and M_epoch, or neither. Discarding '+existing+' input.')
		self.name = name
		self.M = (mu/G) # Total mass of orbiting body and parent, in M_sol

	def printElems(self, deg=False):
		"""
		Print array of orbital elements, in degrees or radians.
		Optional input:
			deg -- output is in degrees if True, else in radians. Default is False
		"""
		if deg:
			print('a = {a:.3f}; e = {e:.3f}; i = {i:.1f} deg; Anode = {anode:.1f} deg; ArgPer = {argper:.1f} deg.'.format(a=self.a, e=self.e, i=self.i*180/np.pi, anode=self.anode*180/np.pi, argper=self.argper*180/np.pi))
		else:
			print('a = {a:.2f}; e = {e:.3f}; i = {i:.1f} rad; Anode = {anode:.1f} rad; ArgPer = {argper:.1f} rad.'.format(a=self.a, e=self.e, i=self.i, anode=self.anode, argper=self.argper))
	
	def printErrs(self, deg=False):
		"""
		Print array of errors in orbital elements, if defined.
		Optional input:
			deg -- output is in degrees if True, else in radians. Default is False
		"""
		try:
			if deg:
				print('d_a = {d_a:.0e}; d_e = {d_e:.0e}; d_i = {d_i:.0e} deg; d_Anode = {d_anode:.0e} deg; d_ArgPer = {d_argper:.0e} deg.'.format(d_a=self.d_a, d_e=self.d_e, d_i=self.d_i*180/np.pi, d_anode=self.d_anode*180/np.pi, d_argper=self.d_argper*180/np.pi))
			else:
				print('d_a = {d_a:.0e}; d_e = {d_e:.0e}; d_i = {d_i:.0e} rad; d_Anode = {d_anode:.0e} rad; d_ArgPer = {d_argper:.0e} rad.'.format(d_a=self.d_a, d_e=self.d_e, d_i=self.d_i, d_anode=self.d_anode, d_argper=self.d_argper))
		except:
			print('Error in Orbit.printErrs(): missing uncertainties.')
			
	def elems(self, deg=False):
		"""
		Return an array of the five fixed orbital elements.
		Optional input:
			deg -- output is in degrees if True, else in radians. Default is False
		"""
		if deg:
			elements = [self.a, self.e, self.i*180/np.pi, self.anode*180/np.pi, self.argper*180/np.pi]
		else:
			elements = [self.a, self.e, self.i, self.anode, self.argper]
		return elements

	def shape2coords(self, f):
		"""
		Give coordinates based on the shape, but not orientation, of the orbit.
		Essentially, gives the position and velocity vectors the orbiting object would have if it were equatorial and prograde, with the periapsis on the +x axis.
		Input:
			f -- true anomaly, in radians
		Output:
			R -- numpy vector; position in ecliptic Cartesian coordinates, in AU
			V -- numpy vector; velocity in ecliptic Cartesian coordinates, in AU/day
		"""
		n = np.sqrt(self.mu/(self.a**3))
		E = f2E(f, self.e)
		x = self.a*(np.cos(E) - self.e)
		y = self.a*np.sqrt(1 - (self.e**2))*np.sin(E)
		vx = - self.a*n*np.sin(E)/(1 - self.e*np.cos(E))
		vy = self.a*n*np.cos(E)*np.sqrt(1 - (self.e**2))/(1 - self.e*np.cos(E))
		R = np.array([x,y,0])
		V = np.array([vx,vy,0])
		return R, V

	def rotate(self, R, V):
		"""
		Rotate the given position and velocity vectors by the orbital elements.
		"""
		rot_argper = np.array([
			[np.cos(self.argper), -np.sin(self.argper), 0],
			[np.sin(self.argper), np.cos(self.argper), 0],
			[0, 0, 1]])
		rot_i = np.array([
			[1, 0, 0],
			[0, np.cos(self.i), -np.sin(self.i)],
			[0, np.sin(self.i), np.cos(self.i)]])
		rot_anode = np.array([
			[np.cos(self.anode), -np.sin(self.anode), 0],
			[np.sin(self.anode), np.cos(self.anode), 0],
			[0, 0, 1]])
		R_rot = rot_anode.dot(rot_i.dot(rot_argper.dot(R)))
		V_rot = rot_anode.dot(rot_i.dot(rot_argper.dot(V)))
		return R_rot, V_rot

	def elems2coords(self, f, deg=False):
		"""
		Given a true anomaly, returns position and velocity vectors for the orbiting body.
		Input:
			f -- true anomaly, in radians
		Optional Input:
			deg -- if True, allows f to be given in degrees. Default False.
		Output:
			R -- numpy vector; position in ecliptic Cartesian coordinates, in AU
			V -- numpy vector; velocity in ecliptic Cartesian coordinates, in AU/day
		"""
		if deg: f=f*np.pi/180
		R, V = self.rotate(*self.shape2coords(f))
		return R, V

	def trueAnomaly(self, JD, epoch=None, M_epoch=None, deg=False):
		"""
		Given a Julian Date, and an epoch date and the mean anomaly at the epoch, return true anomaly at the date.
		Inputs:
			JD -- Julian Date
			epoch -- Julian Date of epoch
			M_epoch -- true anomaly at epoch, in radians
		Optional Input:
			deg -- allows M_epoch to be input as degrees
		Output:
			f -- true anomaly in radians
		"""
		if (epoch is None) & (self.epoch is None):
			print('Error in Orbit.trueAnomaly(): no epoch given')
			return None
		elif (epoch is None):
			epoch = self.epoch

		if (M_epoch is None) & (self.M_epoch is None):
			print('Error in Orbit.trueAnomaly(): no M at epoch given')
			return None
		elif (M_epoch is None):
			M_epoch = self.M_epoch

		if deg: M_epoch = M_epoch*np.pi/180
		n = 2*np.pi/self.T
		M = (M_epoch + n*(JD - epoch))
		E = M2E(M, self.e)
		f = E2f(E, self.e)
		return f

	def predictObs(self, JD, H, G_phot, epoch=None, M_epoch=None, deg=False):
		"""
		Given a date, an absolute magnitude, an epoch with mean anomaly at that epoch, and photometric G parameter, return an Observation object describing the body as seen on that date. Can accept degrees or radians.
		Inputs:
			JD -- Julian Date
			H -- absolute magnitude of the object
			G_phot -- photometric G parameter
			epoch -- Julian Date of epoch
			M_epoch -- true anomaly at epoch, in radians
		Optional Input:
			deg -- allows M_epoch to be input as degrees
		Output:
			obs -- an Observation object of the body at JD
		"""
		if (epoch is None) & (self.epoch is None):
			print('Error in Orbit.predictObs(): no epoch given')
			return None
		elif (epoch is None):
			epoch = self.epoch

		if (M_epoch is None) & (self.M_epoch is None):
			print('Error in Orbit.predictObs(): no M at epoch given')
			return None
		elif (M_epoch is None):
			M_epoch = self.M_epoch

		R = Rvector(JD)
		f = self.trueAnomaly(JD, epoch, M_epoch, deg=deg)
		r_obj = self.elems2coords(f)[0]
		rho = r_obj + R # +, not -, since R is Earth-Sun vector, not Sun_Earth
		RA, dec = EclCart2Eq(rho)
		mag = appMag(H, R, rho, G_phot)
		obs = Observation(JD, RA, dec, mag, deg=False)
		return obs

	def generateEphemeris(self, dates, H, G_phot, epoch=None, M_epoch=None, deg=False):
		"""
		Given an array of dates, an absolute magnitude and a photometric G parameter, and an epoch and a mean anomaly at that epoch, return an array of observations for each date. 
		Inputs:
			dates -- list of Julian Dates
			H -- absolute magnitude of the object
			G_phot -- photometric G parameter
			epoch -- Julian Date of epoch
			M_epoch -- true anomaly at epoch, in radians
		Optional Input:
			deg -- allows M_epoch to be input as degrees
		Output:
			ephemeris -- a numpy array of Observation objects, one for each entry in dates.
		"""
		if (epoch is None) & (self.epoch is None):
			print('Error in Orbit.generateEphemeris(): no epoch given')
			return None
		elif (epoch is None):
			epoch = self.epoch

		if (M_epoch is None) & (self.M_epoch is None):
			print('Error in Orbit.generateEphemeris(): no M at epoch given')
			return None
		elif (M_epoch is None):
			M_epoch = self.M_epoch

		ephemeris = np.zeros_like(dates).tolist() # Needs tolist or it won't accept Observation class
		for i,JD in enumerate(dates):
			ephemeris[i] = self.predictObs(JD, H, G_phot, epoch, M_epoch, deg=deg)
		return np.array(ephemeris)

class Observation:
	"""
	Define an observation given Julian date, coordinates as RA and Dec, and apparent magnitude.
	Note that RA and Dec are stored as radians, not degrees.
	Required initialization inputs:
		JD -- Julian Date of observation
		RA -- right ascension, in degrees
		dec -- declination, in degrees
		mag -- apparent magnitude
	Optional initialization input:
		deg -- if False, allows RA and Dec to be specified in radians. Default is True.
	Additional properties after initialization:
		RES -- Earth-Sun vector at JD in ecliptic Cartesian coordinates, in AU.
	"""
	def __init__(self, JD, RA, dec, mag, deg=True):
		self.JD = JD
		if deg:
			self.RA = RA*np.pi/180
			self.dec = dec*np.pi/180
		else:
			self.RA = RA
			self.dec = dec
		self.RES = Rvector(JD)
		self.mag = mag
	
	def EclCart(self, r=1):
		"""
		Return ecliptic Cartesian coordinates of the observation, given distance.
		Optional Input:
			r -- distance to object. Accepts any units. Default is unit vector.
		Outputs:
			R -- numpy vector; position in ecliptic Cartesian coordinates; same units as r.
		"""
		# Convert spherical to Cartesian
		x_g = r*np.cos(self.dec)*np.cos(self.RA)
		y_g = r*np.cos(self.dec)*np.sin(self.RA)
		z_g = r*np.sin(self.dec)
		# Convert equatorial to ecliptic
		x_h = x_g
		y_h = y_g*np.cos(-obl_earth) - z_g*np.sin(-obl_earth)
		z_h = y_g*np.sin(-obl_earth) + z_g*np.cos(-obl_earth)
		R = np.array([x_h,y_h,z_h])
		return R

	def EqCart(self, r=1):
		"""
		Return equatorial Cartesian coordinates of the observation, given distance.
		Optional Input:
			r -- distance to object. Accepts any units. Default is  unit vector.
		Outputs:
			R -- numpy vector; position in equatorial Cartesian coordinates; same units as r.
		"""
		x_g = r*np.cos(self.dec)*np.cos(self.RA)
		y_g = r*np.cos(self.dec)*np.sin(self.RA)
		z_g = r*np.sin(self.dec)
		return np.array([x_h,y_h,z_h])

def planetPositions(JD):
	"""
	Given a date, returns an array of the mass, position, and velocity of each planet in the solar system.
	Input:
		JD -- Julian date
	Output:
		results -- Numpy array. One planet per row, in the order given in Planets array.
			Row structure: [name, M, R, V]
			name -- planet name as defined in planet's Orbit object
			M -- Planet mass in units of M_sol
			R -- heliocentric ecliptic Cartesian position vector of planet at JD
			V -- heliocentric ecliptic Cartesian velocity vector of planet at JD
	"""
	results = np.zeros((len(Planets), 4), dtype=object)
	for i,planet in enumerate(Planets):
		f = planet.trueAnomaly(JD)
		R, V = planet.elems2coords(f)
		M = planet.M - 1
		results[i] = planet.name, M, R, V
	return results

def orbElems(r, v, mu):
	"""
	Given a position vector, velocity vector, and system gravitational parameter, returns the six orbital elements.
	Not designed to handle unbound orbits.
	Inputs: 
		r -- numpy vector; position in ecliptic Cartesian coordinates. In AU
		v -- numpy vector; velocity in ecliptic Cartesian coordinates. In AU/day
		mu -- gravitational parameter G*(M+m) for orbit. Units AU^3 day^-2
	outputs:
		orb -- Orbit object containing the orbital parameters
		f -- true anomaly at the time of the input vectors. In radians.
	"""
	h = np.cross(r,v,axis=0)
	a = 1/((2/np.sqrt(r.dot(r))) - (v.dot(v)/mu))
	if a >= 0:
		# np.min() fixes floating point error when e~=0
		e = np.sqrt(1 - np.min([h.dot(h)/(mu*a), 1]))
	else:
		e = np.sqrt(1 - h.dot(h)/(mu*a))
	i = np.arctan2(np.sqrt(h[0]**2 + h[1]**2), h[2])

	if (e==0):
		# With no periapsis to measure from, use true longitude instead
		f = np.arctan2(r[1],r[0]) 
		argper = 0
	else:
		f = np.arctan2(r.dot(v)*np.sqrt(h.dot(h)), h.dot(h) - mu*np.sqrt(r.dot(r)))
		argper = np.arctan2(r[2]*np.sqrt(h.dot(h)), r[1]*h[0] - r[0]*h[1]) - f
		if argper <= -np.pi:
			argper += 2*np.pi
		elif argper > np.pi:
			argper -= 2*np.pi

	Anode = 0 if i == 0 else np.arctan2(h[0], -h[1])
	if Anode <= -np.pi: Anode += 2*np.pi
	
	orb = Orbit(a, e, i, Anode, argper, mu)
	return orb, f

def EclCart2Eq(R):
	"""
	Given radius vector in ecliptic cartesian coordinates, return equatorial coordinates in radians
	Input:
		R -- numpy vector; position in ecliptic Cartesian coordinates.
	Output:
		RA -- right ascension, in radians
		dec -- declination, in radians
	"""
	if np.linalg.norm(R) == 0:
		print('Object is at the origin; returning (0,0)')
		return 0,0
	else:
		x_g = R[0]
		y_g = R[1]*np.cos(obl_earth) - R[2]*np.sin(obl_earth)
		z_g = R[1]*np.sin(obl_earth) + R[2]*np.cos(obl_earth)
		RA = np.arctan2(y_g,x_g)
		dec = np.arcsin(z_g/np.linalg.norm(R))
		return RA, dec

def EqCart2Eq(R):
	"""
	Converts from geocentric equatorial Cartesian coordinates to RA and Dec.
	Input:
		R -- numpy vector; position in equatorial Cartesian coordinates.
	Output:
		RA -- right ascension, in radians
		dec -- declination, in radians
	"""
	if np.linalg.norm(R) == 0:
		print('Object is at the origin; returning (0,0)')
		return 0,0
	else:
		RA = np.arctan2(R[1], R[0])
		dec = np.arcsin(R[2]/np.linalg.norm(R))
		return RA, dec

def Rvector(JD):
	"""
	Given Julian Date, return the Earth->Sun vector. Note: NOT Sun->Earth vector!
	Input:
		JD -- Julian Date
	Output:
		R -- numpy vector; Earth->Sun vector in ecliptic Cartesian coordinates. In AU.
	"""
	f = Earth.trueAnomaly(JD)
	R = -Earth.elems2coords(f)[0]
	return R

def spread(x,y,z):
	"""
	Estimate uncertainty by getting spread of three input values.
	"""
	spread = 0.5*(max([x,y,z]) - min([x,y,z]))
	return spread

## Anomaly conversions
def f2E(f, e):
	"""
	Given true anomaly and eccentricity, return eccentric anomaly.
	Anomalies given in radians.
	"""
	return 2*np.arctan2(np.sqrt(1-e)*np.sin(0.5*f), np.sqrt(1+e)*np.cos(0.5*f))

def E2f(E, e):
	"""
	Given eccentric anomaly and eccentricity, return true anomaly.
	Anomalies given in radians.
	"""
	return 2*np.arctan2(np.sqrt(1+e)*np.sin(0.5*E), np.sqrt(1-e)*np.cos(0.5*E))

def M2E(M, e, threshold=1e-12, method='Newton'):
	"""
	Iteratively solve Kepler's Equation backwards for eccentric anomaly.
	Can use Newton's method or fixed-point method.
	Inputs:
		M -- true anomaly, in radians
		e -- eccentricity
	Optional Inputs:
		threshold -- stopping threshold for iterative solver. Default is 1e-12
		method -- choose between "Newton" or "Fixed-point" methods. Default is Newton's Method.
	Output:
		E -- eccentric anomaly, in radians
	"""
	E = M
	deltaE = 1
	if method=='Newton':
		while deltaE > threshold:
			oldE = E
			E = oldE - (oldE - e*np.sin(oldE) - M)/(1 - e*np.cos(oldE))
			deltaE = E-oldE
	else:
		while deltaE > threshold:
			oldE = E
			E = M + e*np.sin(oldE)
			deltaE = E - oldE
	return E

def E2M(E, e):
	"""
	Solve Kepler's Equation forwards for mean anomaly, given eccentric anomaly and eccentricity.
	Anomalies in radians.
	"""
	return E - e*np.sin(E)

## Orbit determination
def position(obs1, obs2, obs3):
	"""
	Given three observations, calculates heliocentric and geocentric positions.
	Inputs:
		obs1, obs2, and obs3 -- three Observation objects
	Outputs:
		rs -- list of heliocentric ecliptic Cartesian position vectors at each observation, in AU.
		rhos -- list of ecliptic Cartesian Earth->Object vectors at each observation, in AU.
	"""
	tau2 = obs3.JD - obs1.JD
	b1 = (obs3.JD - obs2.JD)/tau2
	b3 = (obs2.JD - obs1.JD)/tau2
	print('Sector ratios: b1={b1:.2f};\tb3={b3:.2f}'.format(b1=b1, b3=b3))
	rho1hat = obs1.EclCart()
	rho2hat = obs2.EclCart()
	rho3hat = obs3.EclCart()

	# Cunningham transformation
	xi = rho1hat
	eta_crossprod = np.cross(xi, np.cross(rho3hat, xi))
	eta = eta_crossprod/np.linalg.norm(eta_crossprod)
	zeta = np.cross(xi, eta)
	transform_matrix = np.array([xi, eta, zeta])

	rho1hat_cun = transform_matrix.dot(rho1hat)
	rho2hat_cun = transform_matrix.dot(rho2hat)
	rho3hat_cun = transform_matrix.dot(rho3hat)
	#Check that these match Eq. 5.13
	assert(np.allclose(rho1hat_cun, [1,0,0]))
	assert(np.isclose(rho3hat_cun[2], 0))
	nu2 = rho2hat_cun[2]
	print('Check: nu2 = {:.2e}.'.format(nu2))

	R1 = transform_matrix.dot(obs1.RES)
	R2 = transform_matrix.dot(obs2.RES)
	R3 = transform_matrix.dot(obs3.RES)

	rho2 = (-b1*R1[2] + R2[2] - b3*R3[2])/nu2
	rho3 = (rho2*rho2hat_cun[1] + b1*R1[1] - R2[1] + b3*R3[1])/(b3*rho3hat_cun[1])
	rho1 = (rho2*rho2hat_cun[0] - b3*rho3*rho3hat_cun[0] + b1*R1[0] - R2[0] + b3*R3[0])/b1
	rhos = [rho1*rho1hat, rho2*rho2hat, rho3*rho3hat]

	# Back to heliocentric vectors
	r1 = rho1*rho1hat - obs1.RES
	r2 = rho2*rho2hat - obs2.RES
	r3 = rho3*rho3hat - obs3.RES
	rs = [r1, r2, r3]
	return [rs, rhos]

def vel_f(t, t0, sigma, tau):
	"""Equation 5.21"""
	return 1 - 0.5*sigma*(t-t0)**2 + 0.5*sigma*tau*(t-t0)**3

def vel_g(t, t0, sigma):
	"""Equation 5.22"""
	return (t-t0) - sigma*(t-t0)**3/6

def velocity(obs0, obs1, r0, r1):
	"""
	Given two observations and their respective position vectors, return the velocity vector at the time of the earlier observation.
	Inputs:
		obs0 -- Observation object of first observation
		obs1 -- Observation object of second observation
		r0 -- numpy vector; heliocentric ecliptic Cartesian position vector, for first observation. In AU.
		r1 -- numpy vector; heliocentric ecliptic Cartesian position vector, for second observation. In AU.
	Output:
		v0 -- numpy vector; heliocentric ecliptic Cartesian velocity vector, for first observation. In AU/day.
	"""
	sigma = G/(np.linalg.norm(r0)**3)
	v0 = (r1 - vel_f(obs1.JD, obs0.JD, sigma, 0)*r0)/vel_g(obs1.JD, obs0.JD, sigma)
	for _ in range(4): # Iterate to get tau
		tau = r0.dot(v0)/r0.dot(r0)
		v0 = (r1 - vel_f(obs1.JD, obs0.JD, sigma, tau)*r0)/vel_g(obs1.JD, obs0.JD, sigma)
	return v0

def Gauss(obs1, obs2, obs3):
	"""
	Given three Observation objects, determine orbit of the body being observed.
	Inputs:
		obs1, obs2, and obs3 -- three Observation objects
	Outputs:
		mean_orbit -- An Orbit object, where each element in the mean of the three orbits'.
		orbits -- three Orbit objects, corresponding to the three combinations for velocity().
		f2 -- true anomaly at time of second observation. In radians.
		r2 -- heliocentric ecliptic Cartesian position vector at time of second observation. In AU.
		rho2 -- Earth->Object vector at time of second observation. In AU.
	"""
	[r1, r2, r3],[rho1,rho2,rho3] = position(obs1, obs2, obs3)
	v1_2 = velocity(obs1, obs2, r1, r2)
	v1_3 = velocity(obs1, obs3, r1, r3)
	v2_3 = velocity(obs2, obs3, r2, r3)
	orbit1_2,_ = orbElems(r1, v1_2, G)
	orbit1_3,_ = orbElems(r1, v1_3, G)
	orbit2_3, f2 = orbElems(r2, v2_3, G)
	orbits = [orbit1_2, orbit1_3, orbit2_3]

	mean_orbit = Orbit(*np.mean([orbit1_2.elems(), orbit1_3.elems(), orbit2_3.elems()], axis=0), epoch=obs2.JD)
	mean_orbit.M_epoch = E2M(f2E(f2, mean_orbit.e), mean_orbit.e)
	mean_orbit.d_a = spread(orbit1_2.a, orbit1_3.a, orbit2_3.a)
	mean_orbit.d_e = spread(orbit1_2.e, orbit1_3.e, orbit2_3.e)
	mean_orbit.d_i = spread(orbit1_2.i, orbit1_3.i, orbit2_3.i)
	mean_orbit.d_anode = spread(orbit1_2.anode, orbit1_3.anode, orbit2_3.anode)
	mean_orbit.d_argper = spread(orbit1_2.argper, orbit1_3.argper, orbit2_3.argper)
	return [mean_orbit, orbits, f2, r2, rho2]

## Magnitudes
def appMag(H, R, Del, G_phot):
	"""
	Given absolute magnitude, Sun->Object vector, Earth->Object vector, and photometric G parameter, return the apparent magnitude.
	"""
	alpha = np.arccos(Del.dot(R)/(np.linalg.norm(Del)*np.linalg.norm(R)))
	phi1 = np.exp( -A1*np.tan(alpha*0.5)**B1 )
	phi2 = np.exp( -A2*np.tan(alpha*0.5)**B2 )
	apf = 2.5*np.log10( (1-G_phot)*phi1 + G_phot*phi2) # Eq. 4.6
	m = H + 5*np.log10(np.linalg.norm(Del)*np.linalg.norm(R)) - apf
	return m

def absMag(m, R, Del, G_phot):
	"""
	Given apparent magnitude, Sun->Object vector, Earth->Object vector, and photometric G parameter, return the absolute magnitude.
	"""
	alpha = np.arccos(Del.dot(R)/(np.linalg.norm(Del)*np.linalg.norm(R)))
	phi1 = np.exp( -A1*np.tan(alpha*0.5)**B1 )
	phi2 = np.exp( -A2*np.tan(alpha*0.5)**B2 )
	apf = 2.5*np.log10( (1-G_phot)*phi1 + G_phot*phi2) # Eq. 4.6
	H = m - 5*np.log10(np.linalg.norm(Del)*np.linalg.norm(R)) + apf
	return H

def diameter(H, A):
	"""
	Given absolute magnitude and albedo, return diameter in km
	"""
	D = 1329*(10**(-H/5))/np.sqrt(A)
	return D

## Earth-local
def JD2Angle(JD):
	"""
	Gives angle between first point of Aries and Greenwich meridian at given time, in degrees.
	"""
	t = (JD - 2451545.0)
	T = t/36525
	theta = np.mod(280.46061837 + 360.98564736629*t 
		+ 0.0003879332*(T**2) - ((T**3)/38710000), 360)
	return theta

def EqCart2Cartog(R, JD):
	"""
	Rotates Cartesian coordinate vector from x being along the First Point of Aries to x being along the Greenwich meridian.
	"""
	[xg,yg,zg] = R
	theta = JD2Angle(JD)*np.pi/180
	xc = xg*np.cos(theta) + yg*np.sin(theta)
	yc = yg*np.cos(theta) - xg*np.sin(theta)
	zc = zg
	return np.array([xc, yc, zc])

def Cartog2EqCart(R, JD):
	"""
	Rotates Cartesian coordinate vector from x being along the Greenwich meridian to x being along the First Point of Aries.
	"""
	[xc,yc,zc] = R
	theta = JD2Angle(JD)*np.pi/180
	xg = xc*np.cos(theta) - yc*np.sin(theta)
	yg = yc*np.cos(theta) + xc*np.sin(theta)
	zg = zc
	return np.array([xg, yg, zg])

def Cartog2Geod(R):
	"""
	Converts from cartographic Cartesian coordinate vector to geodetic latitude, longitude, and altitude.
	Uses WGS84 reference ellipsoid.
	All distance units must be in km, and angles are in degrees.
	"""
	[xc,yc,zc] = R
	a = 6378.137
	b = 6356.752314245
	phi = np.arctan2(zc*a**2, np.sqrt(xc**2 + yc**2)*b**2)
	lam = np.arctan2(yc,xc)
	phiprime = np.arctan2(zc, np.sqrt(xc**2 + yc**2))
	rho = np.sqrt(xc**2 + yc**2 + zc**2)/a
	u = np.arctan2(a*zc, np.sqrt(xc**2 + yc**2)*b)
	h = a*(rho*np.cos(phiprime) - np.cos(u))/np.cos(phi)
	return phi*180/np.pi, lam*180/np.pi, h

def Geod2Cartog(phi, lam, h):
	"""
	Converts from geodetic latitude, longitude, and altitude to cartographic Cartesian coordinates.
	Uses WGS84 reference ellipsoid.
	All distance units must be in km, and angles in degrees.
	"""
	phi = phi*np.pi/180
	lam = lam*np.pi/180
	a = 6378.137
	b = 6356.752314245
	esquared = 1 - (b/a)**2
	r = 1/np.sqrt(1 - esquared*np.sin(phi)**2)
	x = (r+h)*np.cos(phi)*np.cos(lam)
	y = (r+h)*np.cos(phi)*np.sin(lam)
	z = ((1 - esquared)*r + h)*np.sin(phi)
	return np.array([x,y,z])


# Planet masses as given in 9206 Assn 3, orbits from http://www.met.rdg.ac.uk/~ross/Astronomy/Planets.html
epoch_planets = 2451545.0  # Epoch of orbital parameters below, in JD.
Mercury = Orbit(a=0.38709893, e=0.20563069, i=7.00487, anode=48.33167, argper=77.45645-48.33167, mu=G*(1 + 1/6023600.0), epoch=epoch_planets, M_epoch = 252.25084-77.45645, deg=True, name='Mercury')
Venus = Orbit(a=0.72333199, e=0.00677323, i=3.39471, anode=76.68069, argper=131.53298-76.68069, mu=G*(1 + 1/408523.71), epoch=epoch_planets, M_epoch = 181.97973-131.53298, deg=True, name='Venus')
Earth = Orbit(a=1.00000011, e=0.01671022, i=0.00005, anode=-11.26064, argper=102.94719-(-11.26064), mu=G*(1 + (1/332946.050895)+(1/27068700.387534)), epoch=epoch_planets, M_epoch = 100.46435-102.94719, deg=True, name='Earth') # Includes mass of Moon
Mars = Orbit(a=1.52366231, e=0.09341233, i=1.85061, anode=49.57854, argper=336.04084-49.57854, mu=G*(1 + 1/3098708.0), epoch=epoch_planets, M_epoch = 355.45332-336.04084, deg=True, name='Mars')
Jupiter = Orbit(a=5.20336301, e=0.04839266, i=1.30530, anode=100.55615, argper=14.75385-100.55615, mu=G*(1 + 1/1047.3486), epoch=epoch_planets, M_epoch = 34.40438-14.75385, deg=True, name='Jupiter')
Saturn = Orbit(a=9.53707032, e=0.05415060, i=2.48446, anode=113.71504, argper=92.43194-113.71504, mu=G*(1 + 1/3497.898), epoch=epoch_planets, M_epoch = 49.94432-92.43194, deg=True, name='Saturn')
Uranus = Orbit(a=19.19126393, e=0.04716771, i=0.76986, anode=74.22988, argper=170.96424-74.22988, mu=G*(1 + 1/22902.98), epoch=epoch_planets, M_epoch = 313.23218-170.96424, deg=True, name='Uranus')
Neptune = Orbit(a=30.06896348, e=0.00858587, i=1.76917, anode=131.72169, argper=44.97135-131.72169, mu=G*(1 + 1/19412.24), epoch=epoch_planets, M_epoch = 304.88003-44.97135, deg=True, name='Neptune')
Pluto = Orbit(a=39.48168677, e=0.24880766, i=17.14175, anode=110.30347, argper=224.06676-110.30347, mu=G*(1 + 1/1.352e8), epoch=epoch_planets, M_epoch = 238.92881-224.06676, deg=True, name='Pluto')
Planets = [Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto]