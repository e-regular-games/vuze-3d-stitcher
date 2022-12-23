function e = collect(f, varargin)

  if (nargout > 1)
    print_usage ();
  endif

  f = sym(f);
  for i = 1:length(varargin)
    varargin{i} = sym(varargin{i});
  endfor

  e = pycall_sympy__ ('return collect(*_ins)', f, varargin{:});

endfunction
