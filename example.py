import arbitrary_domain_convection_diffusion as pde_solver

if __name__ == "__main__":
    model_input_file = 'example/model_inputs.csv'
    pde = pde_solver.PDEObject(model_input_file)
    pde.run_pde()
    pde.write_output_to_file()
