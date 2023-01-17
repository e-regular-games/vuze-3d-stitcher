pkg load symbolic
syms a_l b_l c_l a_r b_r c_r r_l r_r r_d L X Y Z a_pl b_pl c_pl a_pr b_pr c_pr a_d b_d c_d;

m_l = [a_l; b_l; c_l];
m_r = [a_r; b_r; c_r];
m_d = [a_d; b_d; c_d];

p_l = [a_pl; b_pl; c_pl];
p_r = [a_pr; b_pr; c_pr];

eqn = p_l + r_l * m_l + r_d * m_d == p_r + r_r * m_r;
sols = solve(eqn, [r_l r_r r_d]);
