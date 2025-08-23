import json
import os

def extract_gamma_energies():
    """Extract gamma energies from the master isotope database"""
    
    # Load the master database
    with open('master_spectra/isotope_baselines_22.json', 'r') as f:
        data = json.load(f)
    
    isotope_energies = {}
    
    for isotope_name, isotope_data in data['isotopes'].items():
        gamma_energies = isotope_data.get('gamma_energies_keV', [])
        gamma_weights = isotope_data.get('gamma_weights_rel', [])
        category = isotope_data.get('category', 'unknown')
        half_life = isotope_data.get('half_life', 'unknown')
        
        isotope_energies[isotope_name] = {
            'category': category,
            'half_life': half_life,
            'gamma_energies_keV': gamma_energies,
            'gamma_weights_rel': gamma_weights,
            'primary_energy_keV': max(gamma_energies) if gamma_energies else None
        }
        
        print(f"{isotope_name}: {gamma_energies} keV ({category})")
    
    return isotope_energies

if __name__ == "__main__":
    energies = extract_gamma_energies()
    
    # Save to a simplified file
    with open('gamma_energies_extracted.json', 'w') as f:
        json.dump(energies, f, indent=2)
    
    print(f"\nExtracted gamma energies for {len(energies)} isotopes")
