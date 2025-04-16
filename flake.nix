{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-24.11";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowBroken = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          stdenv.cc.cc
          gcc
          uv
          (python310.withPackages (ps: with ps; [ numpy torchWithRocm ]))
        ];
        shellHook = ''
          export UV_PYTHON_PREFERENCE="only-system";
          export UV_PYTHON=${pkgs.python310}
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
        '';
      };
    };
}
