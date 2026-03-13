//! # rkf-macros
//!
//! Proc macros for the RKIField behavior system.
//!
//! - `#[component]` — annotate a struct or enum to register it as an ECS component
//!   with auto-derived serialization, ComponentMeta, and type-erased ComponentEntry.
//! - `#[system(phase = Update)]` — annotate a function to register it as an ECS system
//!   with phase scheduling and dependency ordering.

mod component_helpers;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Ident, ItemEnum, ItemFn, ItemStruct, LitStr, Token,
};

use component_helpers::{
    classify_type, gen_get_field_arm, gen_set_field_arm,
    has_persist, has_serde_skip, is_option_entity,
};

/// Attribute macro for ECS components.
///
/// Derives `Serialize`, `Deserialize`, `Clone`, `Default`, generates `ComponentMeta`
/// implementation, generates type-erased `ComponentEntry` for the gameplay registry,
/// and registers the component via `inventory`.
///
/// # Struct example
/// ```ignore
/// #[component]
/// pub struct Health {
///     pub current: f32,
///     pub max: f32,
/// }
/// ```
///
/// # Enum example
/// ```ignore
/// #[component]
/// pub enum GuardState {
///     Patrolling,
///     Alerted,
///     Chasing,
/// }
/// ```
// ─── Component macro attribute parsing ──────────────────────────────────

/// Parsed attributes from `#[component]` or `#[component(no_default)]`.
struct ComponentAttrs {
    no_default: bool,
}

impl Parse for ComponentAttrs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut no_default = false;
        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "no_default" => no_default = true,
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown component attribute `{other}`, expected `no_default`"),
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(ComponentAttrs { no_default })
    }
}

#[proc_macro_attribute]
pub fn component(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as ComponentAttrs);

    // Try parsing as struct first, then as enum.
    let item2 = item.clone();
    if let Ok(input_struct) = syn::parse::<ItemStruct>(item) {
        match component_impl_struct(attrs, input_struct) {
            Ok(tokens) => tokens.into(),
            Err(err) => err.to_compile_error().into(),
        }
    } else if let Ok(input_enum) = syn::parse::<ItemEnum>(item2) {
        match component_impl_enum(attrs, input_enum) {
            Ok(tokens) => tokens.into(),
            Err(err) => err.to_compile_error().into(),
        }
    } else {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[component] can only be applied to structs or enums",
        )
        .to_compile_error()
        .into()
    }
}

fn component_impl_struct(attrs: ComponentAttrs, mut input: ItemStruct) -> syn::Result<TokenStream2> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    // Build the static name for the fields array.
    let fields_static_name = format_ident!(
        "__{}_FIELDS",
        struct_name_str.to_uppercase()
    );

    // Collect field information from named fields.
    let named_fields = match &input.fields {
        syn::Fields::Named(fields) => &fields.named,
        _ => {
            return Err(syn::Error::new_spanned(
                &input,
                "#[component] only supports structs with named fields",
            ));
        }
    };

    struct FieldInfo {
        name: String,
        ident: Ident,
        field_type_variant: Ident,
        transient: bool,
        persist: bool,
        ty: syn::Type,
    }

    let mut field_infos: Vec<FieldInfo> = Vec::new();

    for field in named_fields {
        let ident = field.ident.as_ref().unwrap().clone();
        let name = ident.to_string();
        let transient = has_serde_skip(&field.attrs);
        let persist = has_persist(&field.attrs);
        let field_type_str = classify_type(&field.ty);
        let field_type_variant = Ident::new(field_type_str, proc_macro2::Span::call_site());

        // Reject #[persist] on transient fields — they should not be saved.
        if persist && transient {
            return Err(syn::Error::new_spanned(
                field,
                format!(
                    "field `{}` has both #[serde(skip)] and #[persist]; \
                     transient fields cannot be persisted",
                    name
                ),
            ));
        }

        field_infos.push(FieldInfo {
            name,
            ident,
            field_type_variant,
            transient,
            persist,
            ty: field.ty.clone(),
        });
    }

    let field_count = field_infos.len();

    // Generate FieldMeta array entries.
    let field_meta_entries: Vec<TokenStream2> = field_infos
        .iter()
        .map(|fi| {
            let name = &fi.name;
            let ft = &fi.field_type_variant;
            let transient = fi.transient;
            let persist = fi.persist;
            quote! {
                rkf_runtime::behavior::FieldMeta {
                    name: #name,
                    field_type: rkf_runtime::behavior::FieldType::#ft,
                    transient: #transient,
                    range: None,
                    default: None,
                    persist: #persist,
                }
            }
        })
        .collect();

    // Generate get_field match arms (non-transient fields with supported types).
    let get_field_arms: Vec<TokenStream2> = field_infos
        .iter()
        .filter(|fi| !fi.transient)
        .filter_map(|fi| gen_get_field_arm(&fi.name, &fi.ident, &fi.ty, struct_name))
        .collect();

    // Generate set_field match arms.
    let set_field_arms: Vec<TokenStream2> = field_infos
        .iter()
        .filter(|fi| !fi.transient)
        .filter_map(|fi| gen_set_field_arm(&fi.name, &fi.ident, &fi.ty, struct_name))
        .collect();

    // Determine derives to add.
    let derive_default = if attrs.no_default {
        quote! {}
    } else {
        quote! { Default, }
    };

    let serde_default_attr = if attrs.no_default {
        quote! {}
    } else {
        quote! { #[serde(default)] }
    };

    // Remove any existing derive attributes and re-add with our extras.
    // We prepend our derives to the struct.
    // To avoid double-deriving, we check existing derives. For simplicity,
    // we add our derives and trust the user not to also derive them manually.

    // Strip existing derives for Serialize, Deserialize, Clone, Default to avoid duplicates.
    for attr in &mut input.attrs {
        if attr.path().is_ident("derive") {
            // We can't easily modify derive contents, so we'll leave them.
            // The user shouldn't manually add these derives on a #[component] struct.
        }
    }

    // Strip #[persist] attributes from fields — they've been consumed above.
    // Also inject serde attributes on Entity/Option<Entity> fields so they can
    // serialize as raw u64 bits (hecs::Entity does not implement Serialize).
    if let syn::Fields::Named(ref mut fields) = input.fields {
        for field in fields.named.iter_mut() {
            field.attrs.retain(|attr| !attr.path().is_ident("persist"));

            let class = classify_type(&field.ty);
            if class == "Entity" {
                if is_option_entity(&field.ty) {
                    // Option<Entity>
                    field.attrs.push(syn::parse_quote! {
                        #[serde(
                            serialize_with = "rkf_runtime::behavior::entity_serde::ser_opt_entity",
                            deserialize_with = "rkf_runtime::behavior::entity_serde::de_opt_entity"
                        )]
                    });
                } else {
                    // Plain Entity
                    field.attrs.push(syn::parse_quote! {
                        #[serde(
                            serialize_with = "rkf_runtime::behavior::entity_serde::ser_entity",
                            deserialize_with = "rkf_runtime::behavior::entity_serde::de_entity"
                        )]
                    });
                }
            }
        }
    }

    // Build the output: original struct with added derives + generated impls.
    let vis = &input.vis;
    let struct_attrs = &input.attrs;
    let struct_fields = &input.fields;
    let generics = &input.generics;

    Ok(quote! {
        #(#struct_attrs)*
        #[derive(serde::Serialize, serde::Deserialize, Clone, #derive_default)]
        #serde_default_attr
        #vis struct #struct_name #generics #struct_fields

        // ─── Generated by #[component] ─────────────────────────────────

        const _: () = {
            static #fields_static_name: [rkf_runtime::behavior::FieldMeta; #field_count] = [
                #(#field_meta_entries),*
            ];

            impl #struct_name {
                #[doc(hidden)]
                pub const FIELDS: &'static [rkf_runtime::behavior::FieldMeta] = &#fields_static_name;
            }
        };

        impl rkf_runtime::behavior::ComponentMeta for #struct_name {
            fn type_name() -> &'static str {
                #struct_name_str
            }

            fn fields() -> &'static [rkf_runtime::behavior::FieldMeta] {
                #struct_name::FIELDS
            }
        }

        inventory::submit! {
            rkf_runtime::behavior::ComponentEntry {
                name: #struct_name_str,
                serialize: |world, entity| {
                    world.get::<&#struct_name>(entity)
                        .ok()
                        .map(|c| ron::to_string(&*c).unwrap())
                },
                deserialize_insert: |world, entity, ron_str| {
                    let c: #struct_name = ron::from_str(ron_str).map_err(|e| e.to_string())?;
                    world.insert_one(entity, c).map_err(|e| e.to_string())?;
                    Ok(())
                },
                remove: |world, entity| {
                    let _ = world.remove_one::<#struct_name>(entity);
                },
                has: |world, entity| {
                    world.get::<&#struct_name>(entity).is_ok()
                },
                get_field: |world, entity, field_name| {
                    let c = world.get::<&#struct_name>(entity)
                        .map_err(|_| format!("entity does not have component '{}'", #struct_name_str))?;
                    match field_name {
                        #(#get_field_arms)*
                        _ => Err(format!("unknown field '{}' on component '{}'", field_name, #struct_name_str)),
                    }
                },
                set_field: |world, entity, field_name, value| {
                    let mut c = world.get::<&mut #struct_name>(entity)
                        .map_err(|_| format!("entity does not have component '{}'", #struct_name_str))?;
                    match field_name {
                        #(#set_field_arms)*
                        _ => return Err(format!("unknown field '{}' on component '{}'", field_name, #struct_name_str)),
                    }
                    Ok(())
                },
                meta: #struct_name::FIELDS,
            }
        }
    })
}

// ─── Component macro implementation (enum) ──────────────────────────────

/// Check if the first variant of an enum is a unit variant (no fields).
fn first_variant_is_unit(input: &ItemEnum) -> bool {
    input
        .variants
        .first()
        .map(|v| matches!(v.fields, syn::Fields::Unit))
        .unwrap_or(false)
}

fn component_impl_enum(attrs: ComponentAttrs, input: ItemEnum) -> syn::Result<TokenStream2> {
    let enum_name = &input.ident;
    let enum_name_str = enum_name.to_string();

    // Determine derives. Default is derived only if:
    // - no_default is NOT set, AND
    // - the first variant is a unit variant
    let can_default = !attrs.no_default && first_variant_is_unit(&input);

    let derive_default = if can_default {
        quote! { Default, }
    } else {
        quote! {}
    };

    let vis = &input.vis;
    let enum_attrs = &input.attrs;
    let generics = &input.generics;

    // Build the static fields array name (empty for enums).
    let fields_static_name = format_ident!(
        "__{}_FIELDS",
        enum_name_str.to_uppercase()
    );

    // Rebuild variants, injecting #[default] on the first unit variant when can_default.
    let variant_tokens: Vec<TokenStream2> = input
        .variants
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let v_attrs = &v.attrs;
            let v_ident = &v.ident;
            let v_fields = &v.fields;
            let v_discriminant = v.discriminant.as_ref().map(|(eq, expr)| quote! { #eq #expr });
            let default_attr = if can_default && i == 0 {
                quote! { #[default] }
            } else {
                quote! {}
            };
            quote! {
                #(#v_attrs)*
                #default_attr
                #v_ident #v_fields #v_discriminant
            }
        })
        .collect();

    Ok(quote! {
        #(#enum_attrs)*
        #[derive(serde::Serialize, serde::Deserialize, Clone, #derive_default)]
        #vis enum #enum_name #generics {
            #(#variant_tokens),*
        }

        // ─── Generated by #[component] (enum) ────────────────────────────

        const _: () = {
            static #fields_static_name: [rkf_runtime::behavior::FieldMeta; 0] = [];

            impl #enum_name {
                #[doc(hidden)]
                pub const FIELDS: &'static [rkf_runtime::behavior::FieldMeta] = &#fields_static_name;
            }
        };

        impl rkf_runtime::behavior::ComponentMeta for #enum_name {
            fn type_name() -> &'static str {
                #enum_name_str
            }

            fn fields() -> &'static [rkf_runtime::behavior::FieldMeta] {
                #enum_name::FIELDS
            }
        }

        inventory::submit! {
            rkf_runtime::behavior::ComponentEntry {
                name: #enum_name_str,
                serialize: |world, entity| {
                    world.get::<&#enum_name>(entity)
                        .ok()
                        .map(|c| ron::to_string(&*c).unwrap())
                },
                deserialize_insert: |world, entity, ron_str| {
                    let c: #enum_name = ron::from_str(ron_str).map_err(|e| e.to_string())?;
                    world.insert_one(entity, c).map_err(|e| e.to_string())?;
                    Ok(())
                },
                remove: |world, entity| {
                    let _ = world.remove_one::<#enum_name>(entity);
                },
                has: |world, entity| {
                    world.get::<&#enum_name>(entity).is_ok()
                },
                get_field: |_world, _entity, field_name| {
                    Err(format!(
                        "field access not supported on enum component '{}' (field '{}')",
                        #enum_name_str, field_name
                    ))
                },
                set_field: |_world, _entity, field_name, _value| {
                    Err(format!(
                        "field access not supported on enum component '{}' (field '{}')",
                        #enum_name_str, field_name
                    ))
                },
                meta: #enum_name::FIELDS,
            }
        }
    })
}

// ─── System macro internals ──────────────────────────────────────────────

/// Parsed attributes from `#[system(phase = Update, after = "a", before = "b")]`.
struct SystemAttrs {
    phase: Ident,
    after: Vec<LitStr>,
    before: Vec<LitStr>,
}

impl Parse for SystemAttrs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut phase: Option<Ident> = None;
        let mut after: Vec<LitStr> = Vec::new();
        let mut before: Vec<LitStr> = Vec::new();

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "phase" => {
                    let value: Ident = input.parse()?;
                    let val_str = value.to_string();
                    if val_str != "Update" && val_str != "LateUpdate" {
                        return Err(syn::Error::new(
                            value.span(),
                            format!(
                                "unknown phase `{val_str}`, expected `Update` or `LateUpdate`"
                            ),
                        ));
                    }
                    phase = Some(value);
                }
                "after" => {
                    after = parse_string_list(input)?;
                }
                "before" => {
                    before = parse_string_list(input)?;
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown attribute `{other}`, expected `phase`, `after`, or `before`"),
                    ));
                }
            }

            // Consume optional trailing comma.
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        let phase = phase.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "`phase` attribute is required")
        })?;

        Ok(SystemAttrs {
            phase,
            after,
            before,
        })
    }
}

/// Parse either a single string literal or a bracketed comma-separated list.
/// e.g. `"foo"` or `["foo", "bar"]`
fn parse_string_list(input: ParseStream) -> syn::Result<Vec<LitStr>> {
    if input.peek(syn::token::Bracket) {
        let content;
        syn::bracketed!(content in input);
        let punct =
            content.parse_terminated(|input: ParseStream| input.parse::<LitStr>(), Token![,])?;
        Ok(punct.into_iter().collect())
    } else {
        let lit: LitStr = input.parse()?;
        Ok(vec![lit])
    }
}

/// Attribute macro for ECS systems.
///
/// Registers the annotated function as a system with phase scheduling
/// and optional dependency ordering.
///
/// # Attributes
///
/// - `phase` (required): `Update` or `LateUpdate`
/// - `after` (optional): system name(s) that must run before this one
/// - `before` (optional): system name(s) that must run after this one
///
/// Both `after` and `before` accept a single string (`after = "foo"`) or a
/// bracketed list (`after = ["foo", "bar"]`).
///
/// # Generated code
///
/// The macro preserves the original function and generates a hidden registration
/// function `__register_system_<name>` that can be called by the gameplay registry
/// to register the system's metadata.
///
/// # Example
/// ```ignore
/// #[system(phase = Update)]
/// fn patrol_system(ctx: &mut SystemContext) {
///     // ...
/// }
///
/// #[system(phase = Update, after = "combat_system")]
/// fn death_system(ctx: &mut SystemContext) {
///     // ...
/// }
///
/// #[system(phase = LateUpdate, after = ["movement", "combat"], before = "cleanup")]
/// fn camera_follow(ctx: &mut SystemContext) {
///     // ...
/// }
/// ```
#[proc_macro_attribute]
pub fn system(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as SystemAttrs);
    let func = parse_macro_input!(item as ItemFn);

    match system_impl(attrs, func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn system_impl(attrs: SystemAttrs, func: ItemFn) -> syn::Result<TokenStream2> {
    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();

    let register_fn_name = Ident::new(
        &format!("__register_system_{fn_name_str}"),
        fn_name.span(),
    );

    let phase_ident = &attrs.phase;

    let after_strs: Vec<String> = attrs.after.iter().map(|l| l.value()).collect();
    let before_strs: Vec<String> = attrs.before.iter().map(|l| l.value()).collect();

    // Generate the static slices for after/before dependencies.
    let after_tokens = if after_strs.is_empty() {
        quote! { &[] }
    } else {
        quote! { &[#(#after_strs),*] }
    };

    let before_tokens = if before_strs.is_empty() {
        quote! { &[] }
    } else {
        quote! { &[#(#before_strs),*] }
    };

    Ok(quote! {
        #func

        #[doc(hidden)]
        #[allow(non_snake_case)]
        pub fn #register_fn_name(registry: &mut rkf_runtime::behavior::GameplayRegistry) {
            registry.register_system(rkf_runtime::behavior::SystemMeta {
                name: #fn_name_str,
                module_path: module_path!(),
                phase: rkf_runtime::behavior::Phase::#phase_ident,
                after: #after_tokens,
                before: #before_tokens,
                fn_ptr: #fn_name as *const (),
            });
        }
    })
}
