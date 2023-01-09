
% load the data.
depth_rotation_data

n = length(features_l);

#L = -0.01140864702939206;
#R_l = [ cos(L) 0 sin(L); 0 1 0; -sin(L) 0 cos(L) ];
#features_l = features_l * R_l + [ 0 0.01 -0.005 ];

[F inliers] = ransacfitfundmatrix(features_l', features_r', 0.000005); #0.000005
length(inliers)

[u s v] = svd(F);

w = [0 -1 0; 1 0 0; 0 0 1];

r = features_r(inliers,:);
l = features_l(inliers,:);

u = 0.06 * u;
Rs = { (u*w'*v') (u*w'*v') (u*w*v') (u*w*v') };
t = u(:,3);
ts = { -t t -t t };

a_l_err = [];
for i = 1:length(Rs)
  
  if sum(diag(Rs{i}) < 0) == 3
    Rs{i} = -Rs{i};
  elseif sum(diag(Rs{i}) < 0) > 0
    a_l_err = [ a_l_err inf ];
    continue
  endif
  err = ( Rs{i} * r' + ts{i} )' - l;
  a_l_err = [ a_l_err sum(sum(err' * err)) ];
end

[_, i] = min(a_l_err);

R = Rs{i}
t = ts{i}

theta_x = atan2(R(3,2), R(3,3))
theta_y = atan2(-R(3,1), sqrt(R(3,2)**2 + R(3,3)**2))
theta_z = atan2(R(2,1), R(1,1))

R_y = [ cos(-theta_y) 0 sin(-theta_y); 0 1 0; -sin(-theta_y) 0 cos(-theta_y) ];
corrected_l = features_l * R_y - (t' .* [0 1 1]);

pkg load symbolic
syms a_l b_l c_l a_r b_r c_r r_l r_r d L X Y Z a_pl b_pl c_pl a_pr b_pr c_pr a_d b_d c_d;

m_l = [a_l; b_l; c_l];
m_r = [a_r; b_r; c_r];
m_d = [a_d; b_d; c_d];
#cross(m_r, m_l);

p_l = [a_pl; b_pl; c_pl];
p_r = [a_pr; b_pr; c_pr];

eqn = p_l + r_l * m_l + d * m_d == p_r + r_r * m_r;
sols = solve(eqn, [r_l r_r d]);

d = zeros(n, 1);
d0 = zeros(n, 1);
r_l = zeros(n, 1);
r_r = zeros(n, 1);
  
warning('off', 'OctSymPy:sym:rationalapprox')
for i = 1:n
  d0(i) = double(subs(sols.d, [X a_l b_l c_l a_r b_r c_r], [0.03 features_l(i,:) features_r(i,:)]));
  d(i) = double(subs(sols.d, [X a_l b_l c_l a_r b_r c_r], [0.03 corrected_l(i,:) features_r(i,:)]));
  r_l(i) = double(subs(sols.r_l, [X a_l b_l c_l a_r b_r c_r], [0.03 corrected_l(i,:) features_r(i,:)]));
  r_r(i) = double(subs(sols.r_r, [X a_l b_l c_l a_r b_r c_r], [0.03 corrected_l(i,:) features_r(i,:)]));
end
warning('on', 'OctSymPy:sym:rationalapprox')

fprintf('initial intersect with 0.04: %i\n', sum(abs(d0) < 0.04));
fprintf('corrected intersect with 0.04: %i\n', sum(abs(d) < 0.04));

