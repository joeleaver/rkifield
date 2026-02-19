mod showcase;

fn main() {
    env_logger::init();
    println!("rkf-game: RKIField example game / playground");

    let scene = showcase::build_showcase();
    println!("Showcase scene loaded:");
    println!("  Entities: {}", scene.entity_count());
    println!("  Lights: {}", scene.light_count());
    println!("  Particle emitters: {}", scene.particle_emitter_count());
    println!("  Spawn point: {:?}", scene.spawn_point);
    println!(
        "  Environment: fog_density={}, cloud_coverage={}, sun_intensity={}",
        scene.environment.fog_density,
        scene.environment.cloud_coverage,
        scene.environment.sun_intensity
    );
}
