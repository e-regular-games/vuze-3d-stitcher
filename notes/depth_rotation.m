depth_rotation_data

n = length(features_l);
 
pkg load symbolic
syms a_l b_l c_l a_r b_r c_r r_l r_r d L X Y Z z

SOLVE = false

if SOLVE
  R_l = subs([ cos(L) z sin(L); z 1 z; -sin(L) z cos(L) ], z, 0);
  
  m_l = R_l * [a_l; b_l; c_l];
  m_r = [a_r; b_r; c_r];
  m_d = cross(m_r, m_l);

  p_l = [-X; Y; Z];
  p_r = [X; Y; Z];

  eqn = p_l + r_l * m_l + d * m_d == p_r + r_r * m_r;

  sols = solve(eqn, [r_l r_r d]);
  dd = sols.d * sols.d;
  d_dd = diff(dd, L);

  d_dd_num = numden(simplify(d_dd));
  d_dd_num_exp = expand(numden(d_dd_num));
  d_dd_num_exp = subs(d_dd_num_exp, [cos(2*L) sin(2*L)], [2*cos(L)*cos(L)-1 2*sin(L)*cos(L)]);
  d_dd_num_exp = expand(d_dd_num_exp);

  L_sols = solve(d_dd_num_exp == 0, L);

  sols = struct(
    'L', L_sols,
    'r_l', sols.r_l,
    'r_r', sols.r_r,
    'd', sols.d);
else
  L_sols = [
    -2*atan((a_l*b_r - sqrt(a_l**2*b_r**2 - b_l**2*c_r**2 + b_r**2*c_l**2))/(b_l*c_r + b_r*c_l)),
    -2*atan((a_l*b_r + sqrt(a_l**2*b_r**2 - b_l**2*c_r**2 + b_r**2*c_l**2))/(b_l*c_r + b_r*c_l))];
    
  r_l_sols = 2*X*a_l*a_r*c_r*sin(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) + 2*X*a_l*b_r**2*cos(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2+ b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) + 2*X*a_l*c_r**2*cos(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L)+ 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) +b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) - 2*X*a_r*b_l*b_r/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) - 2*X*a_r*c_l*c_r*cos(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) +a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) +b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) + 2*X*b_r**2*c_l*sin(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) + 2*X*c_l*c_r**2*sin(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2);
  r_r_sols = -2*X*a_l**2*a_r*sin(L)**2/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) - 2*X*a_l**2*c_r*sin(L)*cos(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) + 4*X*a_l*a_r*c_l*sin(L)*cos(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) + 2*X*a_l*b_l*b_r*cos(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) +a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) +b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) - 2*X*a_l*c_l*c_r*sin(L)**2/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) + 2*X*a_l*c_l*c_r*cos(L)**2/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) - 2*X*a_r*b_l**2/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) - 2*X*a_r*c_l**2*cos(L)**2/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) + 2*X*b_l*b_r*c_l*sin(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) + 2*X*c_l**2*c_r*sin(L)*cos(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2+ a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2);
  d_sols = -2*X*a_l*b_r*sin(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 +a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) - 2*X*b_l*c_r/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2) + 2*X*b_r*c_l*cos(L)/(a_l**2*a_r**2*sin(L)**2 + 2*a_l**2*a_r*c_r*sin(L)*cos(L) + a_l**2*b_r**2*sin(L)**2 + a_l**2*b_r**2*cos(L)**2 + a_l**2*c_r**2*cos(L)**2 - 2*a_l*a_r**2*c_l*sin(L)*cos(L) - 2*a_l*a_r*b_l*b_r*cos(L) + 2*a_l*a_r*c_l*c_r*sin(L)**2 - 2*a_l*a_r*c_l*c_r*cos(L)**2 + 2*a_l*b_l*b_r*c_r*sin(L) + 2*a_l*c_l*c_r**2*sin(L)*cos(L) + a_r**2*b_l**2 + a_r**2*c_l**2*cos(L)**2 - 2*a_r*b_l*b_r*c_l*sin(L) - 2*a_r*c_l**2*c_r*sin(L)*cos(L) + b_l**2*c_r**2 - 2*b_l*b_r*c_l*c_r*cos(L) + b_r**2*c_l**2*sin(L)**2 + b_r**2*c_l**2*cos(L)**2 + c_l**2*c_r**2*sin(L)**2);
  
  sols = struct(
    'L', L_sols,
    'r_l', r_l_sols,
    'r_r', r_r_sols,
    'd', d_sols);  
end

function [d, r_l, r_r] = calculate_radii(sols, l, r, L_a)
  syms L X a_l b_l c_l a_r b_r c_r

  n = length(L_a);
  d = zeros(n, 1);
  r_l = zeros(n, 1);
  r_r = zeros(n, 1);
  
  warning('off', 'OctSymPy:sym:rationalapprox')
  for i = 1:n
    d(i) = double(subs(sols.d, [L X a_l b_l c_l a_r b_r c_r], [L_a(i) 0.03 l(i,:) r(i,:)]));
    r_l(i) = double(subs(sols.r_l, [L X a_l b_l c_l a_r b_r c_r], [L_a(i) 0.03 l(i,:) r(i,:)]));
    r_r(i) = double(subs(sols.r_r, [L X a_l b_l c_l a_r b_r c_r], [L_a(i) 0.03 l(i,:) r(i,:)]));
  end
  warning('on',  'OctSymPy:sym:rationalapprox')

end

L_calc = zeros(n, 2);

warning('off', 'OctSymPy:sym:rationalapprox')
for i = 1:n
  L_calc(i,:) = double(subs(sols.L, [X a_l b_l c_l a_r b_r c_r], [0.03 sample_l(i,:) sample_r(i,:)]))';
end
warning('on',  'OctSymPy:sym:rationalapprox')

d0 = zeros(n, 1);
r_l0 = zeros(n, 1);
r_r0 = zeros(n, 1);

[d0, r_l0, r_r0] = calculate_radii(sols, sample_l, sample_r, zeros(n,1));

d1 = zeros(n, 2);
r_l1 = zeros(n, 2);
r_r1 = zeros(n, 2);

[d1(:,1), r_l1(:,1), r_r1(:,1)] = calculate_radii(sols, sample_l, sample_r, L_calc(:,1));
[d1(:,2), r_l1(:,2), r_r1(:,2)] = calculate_radii(sols, sample_l, sample_r, L_calc(:,2));


[~, minL] = min(L_calc, [], 2);
Ls_min = L_calc(sub2ind(size(L_calc), (1:n)', minL));
Ls_min(imag(Ls_min) != 0) = 0;
Ls_min(abs(ds0) > 0.04) = 0;

sum(Ls_min) / sum(Ls_min != 0)
  
  


#d_dd_L_num_exp = collect(d_dd_L_num_exp, [cos(L) sin(L) cos(L)*sin(L) cos(L)*cos(L)*sin(L) cos(L)*sin(L)*sin(L) cos(L)*cos(L)*cos(L)*sin(L) cos(L)*cos(L)*sin(L)*sin(L) cos(L)*sin(L)*sin(L)*sin(L)])
#sols_L = solve(dd_L == 0, L)

#double(subs(sols_L(1), [X a_l b_l c_l a_r b_r c_r], [0.03 sample_l(:,4)' sample_r(:,4)']))

#delta_dd_rho_1 = simplify(diff(dd, rho_1))
#delta_dd_rho_2 = simplify(diff(dd, rho_2))

#num_delta_dd_rho_1 = collect(expand(numden(delta_dd_rho_1)), [rho_1 rho_2 rho_1*rho_2 rho_1*rho_2*rho_2 rho_1*rho_1*rho_2])
#num_delta_dd_rho_2 = collect(expand(numden(delta_dd_rho_2)), [rho_1 rho_2 rho_1*rho_2 rho_1*rho_2*rho_2 rho_1*rho_1*rho_2])

#constraint_rho_1 = solve([rho_1*rho_1 + rho_2*rho_2 == 1], rho_1) #provides 2 solutions

#min_rho_dd_2_1 = solve(subs(num_delta_dd_rho_1 == 0, rho_1, constraint_rho_1(1)), rho_2)
#min_rho_dd_2_2 = solve(subs(num_delta_dd_rho_1 == 0, rho_1, constraint_rho_1(2)), rho_2)

#rho_min_dd = solve([num_delta_dd_rho_1 == 0; num_delta_dd_rho_2 == 0], [rho_1 rho_2])
#rho_min_dd = [(a_l*b_r*rho_2 + b_l*c_r)/(b_r*c_l); b_l*(-a_l*c_r + a_r*c_l)/(b_r*(a_l**2 + c_l**2)); b_l*(-a_l*c_r + a_r*c_l)/(b_r*(a_l**2 + c_l**2))];


# since we want to eventually find the point where the derivative is 0, we can use only the numerator.
#ndd_rho_1 = collect(expand(numden(delta_dd_rho_1)), [rho_1 rho_2 rho_1*rho_2 rho_1*rho_2*rho_2 rho_1*rho_1*rho_2])
#ndd_rho_2 = collect(expand(numden(delta_dd_rho_2)), [rho_1 rho_2 rho_1*rho_2 rho_1*rho_2*rho_2 rho_1*rho_1*rho_2])

#rho = solve([ndd_rho_1 == 0; ndd_rho_2 == 0], [rho_1 rho_2])

#rhos = ones(n, 2)
#for i = 1:n
#  rhos(i, 1) = double(subs(rho_min_1, [X a_l b_l c_l a_r b_r c_r], [0.03 sample_l(:,i)' sample_r(:,i)']));
#  rhos(i, 2) = double(subs(rho_min_2, [X a_l b_l c_l a_r b_r c_r], [0.03 sample_l(:,i)' sample_r(:,i)']));
#end


#  vpa(subs(sols_L, [a_l b_l c_l a_r b_r c_r], [sample_l(:,i)' sample_r(:,i)']));


  
# try to calculate

